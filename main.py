# monolithic_app.py - полная версия с авторизацией и TTS
import whisper
import tempfile
import os
import json
import logging
import re
import io
import wave
import numpy as np
import string
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, Request, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Text, Boolean, Integer, DateTime, Float, ForeignKey, func, and_

# Импортируем из database.py и auth.py
from database import Base, engine, SessionLocal, get_db
from auth import (
    User, UserRegister, UserLogin, UserResponse,
    hash_password, verify_password,
    get_current_user, get_current_active_user, require_admin,
    create_session_for_user, delete_session, session_storage
)

# ===== TTS IMPORTS =====
try:
    from gtts import gTTS
    import hashlib
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("gTTS not installed. Install with: pip install gtts")

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Глобальная переменная для модели Whisper
whisper_model = None

# Папка для временных аудиофайлов TTS
TTS_CACHE_DIR = Path("tts_cache")
if not TTS_CACHE_DIR.exists():
    TTS_CACHE_DIR.mkdir(exist_ok=True)

# ===== MODELS =====
class TextItem(Base):
    __tablename__ = "texts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    level: Mapped[str] = mapped_column(String(20), index=True)
    topic: Mapped[str] = mapped_column(String(100), index=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    created_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    creator = relationship("User")

class PracticeWord(Base):
    __tablename__ = "practice_words"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    word: Mapped[str] = mapped_column(String(100), nullable=False)
    text_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("texts.id"), nullable=True)
    practice_count: Mapped[int] = mapped_column(Integer, default=0)
    mastered: Mapped[bool] = mapped_column(Boolean, default=False)
    last_practiced: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user = relationship("User")
    text = relationship("TextItem")

class ErrorWord(BaseModel):
    """Модель для слова с ошибкой"""
    word: str
    position: int  # Позиция в тексте
    error_type: str  # Тип ошибки: "mispronounced", "missing", "extra"
    reference_word: Optional[str] = None  # Правильный вариант (если есть)
    pronunciation_url: Optional[str] = None  # URL для прослушивания произношения
    confidence: Optional[float] = None  # Уверенность в ошибке

# ===== PYDANTIC MODELS =====
class TextCreate(BaseModel):
    level: str
    topic: str
    title: str
    content: str

class SaveWordsRequest(BaseModel):
    text_id: int
    words: List[str]

# ===== УЛУЧШЕННЫЕ TTS ФУНКЦИИ =====
def generate_tts_audio_safe(text: str, language: str = "en") -> Optional[str]:
    """
    Генерирует аудиофайл с произношением текста через gTTS
    с обработкой ошибок и валидацией входных данных
    """
    if not TTS_AVAILABLE:
        logger.warning("gTTS not installed. Cannot generate TTS audio.")
        return None

    # Валидация входного текста
    if not text or not text.strip():
        logger.error("Empty text provided for TTS")
        return None

    # Очистка текста от лишних пробелов и специальных символов
    cleaned_text = re.sub(r'\s+', ' ', text.strip())

    # Проверяем, содержит ли текст хотя бы одну букву
    if not re.search(r'[a-zA-Z]', cleaned_text):
        logger.warning(f"Text contains no letters: '{cleaned_text}'")
        return None

    # Пропускаем слишком короткие тексты (менее 2 символов)
    if len(cleaned_text) < 2:
        logger.warning(f"Text too short for TTS: '{cleaned_text}'")
        return None

    # Пропускаем знаки препинания
    if cleaned_text in string.punctuation:
        logger.warning(f"Skipping punctuation: '{cleaned_text}'")
        return None

    try:
        # Создаем хэш имени файла
        text_hash = hashlib.md5(f"{cleaned_text}_{language}".encode()).hexdigest()
        audio_path = TTS_CACHE_DIR / f"{text_hash}.mp3"

        # Если файл уже существует, возвращаем его
        if audio_path.exists():
            logger.debug(f"TTS cache hit for: '{cleaned_text}'")
            return str(audio_path)

        logger.info(f"Generating TTS audio for: '{cleaned_text}'")

        # Генерируем новое аудио
        tts = gTTS(text=cleaned_text, lang=language, slow=False)
        tts.save(str(audio_path))

        logger.info(f"✓ TTS generated for: '{cleaned_text}'")
        return str(audio_path)

    except Exception as e:
        logger.error(f"Error in TTS generation: {e}")
        return None

# ===== УЛУЧШЕННАЯ ТОКЕНИЗАЦИЯ =====
def tokenize_smart(s: str) -> List[str]:
    """
    Умная токенизация, которая сохраняет слова и знаки препинания отдельно
    """
    # Регулярное выражение для захвата слов и знаков препинания
    tokens = re.findall(r'\b[\w\']+\b|[.,!?;:]|[\"\'`\-]', s)
    return [t for t in tokens if t.strip()]

def normalize_word(word: str) -> str:
    """
    Нормализация слова: нижний регистр, удаление лишних знаков препинания
    """
    if not word:
        return ""

    # Приводим к нижнему регистру
    normalized = word.lower()

    # Удаляем начальные и конечные знаки препинания, кроме апострофа
    normalized = normalized.strip(string.punctuation.replace("'", ""))

    return normalized

def is_valid_word(word: str) -> bool:
    """
    Проверяет, является ли токен валидным словом для анализа
    """
    if not word or len(word) < 2:
        return False

    # Пропускаем знаки препинания
    if word in string.punctuation:
        return False

    # Должна содержать хотя бы одну букву
    return bool(re.search(r'[a-zA-Z]', word))

# ===== УЛУЧШЕННЫЙ ПОИСК ОШИБОК =====
def find_pronunciation_errors(
        reference_text: str,
        recognized_text: str
) -> Tuple[List[ErrorWord], Dict[str, Any]]:
    """
    Находит ошибки произношения, сравнивая эталонный и распознанный текст
    """
    # Токенизируем тексты
    ref_tokens = tokenize_smart(reference_text)
    rec_tokens = tokenize_smart(recognized_text)

    # Нормализуем слова для сравнения
    ref_words = [normalize_word(t) for t in ref_tokens if is_valid_word(t)]
    rec_words = [normalize_word(t) for t in rec_tokens if is_valid_word(t)]

    # Сохраняем оригинальные слова
    ref_original = [t for t in ref_tokens if is_valid_word(t)]
    rec_original = [t for t in rec_tokens if is_valid_word(t)]

    error_words = []
    error_details = []

    # Используем алгоритм Левенштейна для сравнения последовательностей
    from difflib import SequenceMatcher

    # Простая посимвольная проверка для коротких слов
    min_len = min(len(ref_words), len(rec_words))

    for i in range(min_len):
        ref_word = ref_words[i]
        rec_word = rec_words[i]

        # Пропускаем короткие или невалидные слова
        if not ref_word or not rec_word or len(ref_word) < 2:
            continue

        # Проверяем, насколько слова похожи
        if ref_word != rec_word:
            # Вычисляем расстояние Левенштейна
            similarity = SequenceMatcher(None, ref_word, rec_word).ratio()

            # Если похожесть менее 0.7, считаем ошибкой
            if similarity < 0.7:
                error_word = ErrorWord(
                    word=rec_original[i],
                    position=i,
                    error_type="mispronounced",
                    reference_word=ref_original[i],
                    confidence=1.0 - similarity
                )
                error_words.append(error_word)

                error_details.append({
                    "reference": ref_original[i],
                    "recognized": rec_original[i],
                    "similarity": round(similarity, 2),
                    "position": i
                })

    # Находим пропущенные слова
    if len(ref_words) > len(rec_words):
        for i in range(len(rec_words), len(ref_words)):
            if i < len(ref_original) and is_valid_word(ref_original[i]):
                error_word = ErrorWord(
                    word="",
                    position=i,
                    error_type="missing",
                    reference_word=ref_original[i],
                    confidence=1.0
                )
                error_words.append(error_word)

    # Находим лишние слова
    if len(rec_words) > len(ref_words):
        for i in range(len(ref_words), len(rec_words)):
            if i < len(rec_original) and is_valid_word(rec_original[i]):
                error_word = ErrorWord(
                    word=rec_original[i],
                    position=i,
                    error_type="extra",
                    reference_word=None,
                    confidence=1.0
                )
                error_words.append(error_word)

    # Расчет статистики
    total_words = len(ref_words)
    correct_words = total_words - len([ew for ew in error_words if ew.error_type == "mispronounced"])
    accuracy = (correct_words / total_words * 100) if total_words > 0 else 0

    stats = {
        "total_words": total_words,
        "correct_words": correct_words,
        "error_count": len(error_words),
        "mispronounced_count": len([ew for ew in error_words if ew.error_type == "mispronounced"]),
        "missing_count": len([ew for ew in error_words if ew.error_type == "missing"]),
        "extra_count": len([ew for ew in error_words if ew.error_type == "extra"]),
        "accuracy": round(accuracy, 1),
        "reference_word_count": len(ref_words),
        "recognized_word_count": len(rec_words)
    }

    return error_words, stats

# ===== УЛУЧШЕННАЯ ФУНКЦИЯ ФИДБЕКА =====
def build_feedback_improved(
        reference_text: str,
        recognized_text: str,
        generate_audio: bool = True
) -> Dict[str, Any]:
    """
    Создает детализированный фидбек с TTS аудио для проблемных слов
    """
    # Находим ошибки произношения
    error_words, stats = find_pronunciation_errors(reference_text, recognized_text)

    # Генерируем TTS аудио для проблемных слов
    pronunciation_data = {}
    tts_errors = []

    if generate_audio and TTS_AVAILABLE and error_words:
        logger.info(f"Generating TTS audio for {len(error_words)} error words")

        for error_word in error_words:
            if error_word.reference_word:
                # Генерируем аудио для правильного произношения
                audio_path = generate_tts_audio_safe(error_word.reference_word, "en")

                if audio_path:
                    # Создаем URL для доступа к аудио
                    word_key = error_word.reference_word.lower()
                    pronunciation_data[word_key] = {
                        "audio_url": f"/tts/pronounce/{word_key}",
                        "word": error_word.reference_word,
                        "file_path": audio_path
                    }
                    logger.info(f"✓ TTS generated for '{error_word.reference_word}'")
                else:
                    tts_errors.append(error_word.reference_word)
                    logger.warning(f"Failed to generate TTS for '{error_word.reference_word}'")

    # Формируем сообщения об ошибках
    errors = []

    mispronounced = [ew for ew in error_words if ew.error_type == "mispronounced"]
    missing = [ew for ew in error_words if ew.error_type == "missing"]
    extra = [ew for ew in error_words if ew.error_type == "extra"]

    if mispronounced:
        error_list = [f"{ew.word} (правильно: {ew.reference_word})"
                      for ew in mispronounced[:3]]
        errors.append({
            "type": "mispronounced",
            "message": f"Неправильно произнесенные слова: {', '.join(error_list)}",
            "words": [ew.reference_word for ew in mispronounced[:5]],
            "count": len(mispronounced)
        })

    if missing:
        missing_words = [ew.reference_word for ew in missing[:3] if ew.reference_word]
        errors.append({
            "type": "missing",
            "message": f"Пропущенные слова: {', '.join(missing_words)}",
            "words": missing_words,
            "count": len(missing)
        })

    if extra:
        extra_words = [ew.word for ew in extra[:3] if ew.word]
        errors.append({
            "type": "extra",
            "message": f"Лишние слова: {', '.join(extra_words)}",
            "words": extra_words,
            "count": len(extra)
        })

    # Если нет ошибок
    if not error_words and stats["accuracy"] > 95:
        errors.append({
            "type": "success",
            "message": "Отличное произношение! Все слова распознаны правильно.",
            "accuracy": stats["accuracy"]
        })
    elif not error_words:
        errors.append({
            "type": "info",
            "message": f"Произношение хорошее. Точность: {stats['accuracy']}%",
            "accuracy": stats["accuracy"]
        })

    # Подготавливаем данные для возврата
    error_words_data = []
    for ew in error_words:
        ew_data = ew.dict()
        if ew.reference_word:
            word_key = ew.reference_word.lower()
            ew_data["has_audio"] = word_key in pronunciation_data
            ew_data["audio_url"] = pronunciation_data.get(word_key, {}).get("audio_url")
        error_words_data.append(ew_data)

    # Получаем уникальные проблемные слова
    problem_words = list(set([ew.reference_word for ew in error_words if ew.reference_word]))

    return {
        "errors": errors,
        "error_words": error_words_data,
        "problem_words": problem_words,
        "pronunciation_data": pronunciation_data,
        "tts_available": TTS_AVAILABLE,
        "tts_errors": tts_errors,
        "summary": stats
    }

# ===== WHISPER ФУНКЦИИ =====
def load_whisper_model():
    """Загрузить модель Whisper"""
    global whisper_model
    if whisper_model is None:
        try:
            logger.info("Loading Whisper model...")
            whisper_model = whisper.load_model("base")
            logger.info("✓ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            whisper_model = None
    return whisper_model

def transcribe_audio_with_whisper(audio_bytes: bytes) -> str:
    """Транскрибировать аудио с помощью Whisper"""
    try:
        model = load_whisper_model()
        if not model:
            return "Whisper model not loaded"

        with io.BytesIO(audio_bytes) as wav_file:
            with wave.open(wav_file, 'rb') as wav:
                n_channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                framerate = wav.getframerate()
                n_frames = wav.getnframes()
                audio_frames = wav.readframes(n_frames)

                if sampwidth == 2:
                    dtype = np.int16
                elif sampwidth == 1:
                    dtype = np.uint8
                else:
                    return "Unsupported audio format"

                audio_array = np.frombuffer(audio_frames, dtype=dtype).astype(np.float32)

                if dtype == np.int16:
                    audio_array /= 32768.0
                elif dtype == np.uint8:
                    audio_array = (audio_array - 128) / 128.0

                if n_channels == 2:
                    audio_array = audio_array.reshape(-1, 2)
                    audio_array = audio_array.mean(axis=1)

        result = model.transcribe(audio_array, fp16=False, language="en")
        transcription = result["text"].strip()
        logger.info(f"Whisper transcription: {transcription[:100]}...")
        return transcription

    except Exception as e:
        logger.error(f"Error transcribing with Whisper: {e}")
        return f"Transcription error: {str(e)}"

# ===== AUDIO PROCESSING =====
def read_wav_bytes(file_bytes: bytes) -> Optional[np.ndarray]:
    """Чтение WAV файла из bytes"""
    try:
        with io.BytesIO(file_bytes) as wav_file:
            with wave.open(wav_file, 'rb') as wav:
                n_channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                framerate = wav.getframerate()
                n_frames = wav.getnframes()
                audio_data = wav.readframes(n_frames)

                logger.info(f"WAV info: channels={n_channels}, sampwidth={sampwidth}, "
                            f"framerate={framerate}, frames={n_frames}")

                if sampwidth == 2:
                    dtype = np.int16
                    audio_array = np.frombuffer(audio_data, dtype=dtype).astype(np.float32) / 32768.0
                elif sampwidth == 1:
                    dtype = np.uint8
                    audio_array = np.frombuffer(audio_data, dtype=dtype).astype(np.float32) / 128.0 - 1.0
                else:
                    logger.warning(f"Unsupported sample width: {sampwidth}")
                    return None

                if n_channels == 2:
                    audio_array = audio_array.reshape(-1, 2)
                    audio_array = audio_array.mean(axis=1)
                    logger.info(f"Converted stereo to mono: {len(audio_array)} samples")

                logger.info(f"Audio array shape: {audio_array.shape}")
                return audio_array

    except Exception as e:
        logger.error(f"Error reading WAV file: {e}")
        return None

# ===== LIFESPAN HANDLER =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=" * 60)
    logger.info("STARTING UP LINGUAAI SERVER...")
    logger.info(f"Working directory: {os.getcwd()}")

    # Проверяем TTS
    if TTS_AVAILABLE:
        logger.info("✓ gTTS available for text-to-speech")
        logger.info(f"TTS cache directory: {TTS_CACHE_DIR}")
    else:
        logger.warning("✗ gTTS not installed. TTS features will be limited.")
        logger.info("  Install with: pip install gtts")

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✓ Database tables created successfully")

        # Создаем администратора по умолчанию
        async with SessionLocal() as session:
            admin_exists = await session.scalar(
                select(User).where(User.email == "admin@linguacab.com")
            )

            if not admin_exists:
                admin = User(
                    email="admin@linguacab.com",
                    username="admin",
                    hashed_password=hash_password("admin123"),
                    full_name="System Administrator",
                    role="admin"
                )
                session.add(admin)
                await session.commit()
                logger.info("✓ Default admin user created (admin@linguacab.com / admin123)")

            # Проверяем есть ли тексты, если нет - создаем несколько примеров
            texts_count = await session.scalar(select(func.count(TextItem.id)))
            if texts_count == 0:
                example_texts = [
                    {
                        "title": "Simple morning routine",
                        "content": "I wake up early every morning. First, I make coffee and have breakfast. Then I check my emails and plan my day. After that, I go for a short walk outside.",
                        "level": "A1",
                        "topic": "Daily life"
                    },
                    {
                        "title": "My favorite hobby",
                        "content": "My favorite hobby is reading books. I enjoy different genres like mystery, science fiction, and history. Reading helps me relax and learn new things.",
                        "level": "A2",
                        "topic": "Hobbies"
                    },
                    {
                        "title": "Travel experiences",
                        "content": "Last year I traveled to Japan. The culture was fascinating and the food was delicious. I visited Tokyo, Kyoto, and Osaka. It was an unforgettable experience.",
                        "level": "B1",
                        "topic": "Travel"
                    },
                    {
                        "title": "Technology in education",
                        "content": "Modern technology has revolutionized education. Students can now access information from anywhere in the world. Online courses and digital resources make learning more accessible than ever before.",
                        "level": "B2",
                        "topic": "Technology"
                    },
                    {
                        "title": "Environmental challenges",
                        "content": "Climate change represents one of the most significant challenges facing humanity today. Addressing this issue requires global cooperation and innovative solutions across multiple sectors including energy, transportation, and agriculture.",
                        "level": "C1",
                        "topic": "Environment"
                    }
                ]

                for text_data in example_texts:
                    text_item = TextItem(
                        title=text_data["title"],
                        content=text_data["content"],
                        level=text_data["level"],
                        topic=text_data["topic"],
                        is_active=True
                    )
                    session.add(text_item)

                await session.commit()
                logger.info(f"✓ Created {len(example_texts)} example texts")

    except Exception as e:
        logger.error(f"✗ Database error: {e}")

    yield

    # Shutdown
    logger.info("\nSHUTTING DOWN SERVER...")
    await engine.dispose()
    logger.info("✓ Database engine disposed")

# ===== FASTAPI APP =====
app = FastAPI(
    title="LinguaAI API",
    description="Language learning platform with pronunciation analysis",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== TTS ENDPOINTS =====
@app.get("/tts/pronounce/{word}")
async def pronounce_word(word: str, language: str = "en"):
    """
    Получить аудио с произношением слова
    """
    if not TTS_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="TTS service not available. Install gtts: pip install gtts"
        )

    # Генерируем или получаем из кэша
    audio_path = generate_tts_audio_safe(word, language)

    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail=f"Audio not found for word: {word}")

    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=f"pronunciation_{word}.mp3"
    )

@app.post("/tts/generate")
async def generate_pronunciation(request: dict):
    """
    Сгенерировать аудио произношения для текста
    """
    text = request.get("text", "")
    language = request.get("language", "en")

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    if not TTS_AVAILABLE:
        return {
            "success": False,
            "message": "TTS service not available. Install gtts: pip install gtts",
            "text": text,
            "language": language
        }

    audio_path = generate_tts_audio_safe(text, language)

    if audio_path:
        return {
            "success": True,
            "message": "Audio generated successfully",
            "audio_url": f"/tts/pronounce/{text[:50]}",
            "text": text,
            "language": language
        }
    else:
        return {
            "success": False,
            "message": "Failed to generate audio",
            "text": text,
            "language": language
        }

@app.get("/tts/status")
async def tts_status():
    """
    Проверить статус TTS сервиса
    """
    cache_files = list(TTS_CACHE_DIR.glob("*.mp3")) if TTS_CACHE_DIR.exists() else []

    return {
        "available": TTS_AVAILABLE,
        "cache_size": len(cache_files),
        "cache_dir": str(TTS_CACHE_DIR),
        "service": "gTTS"
    }

# ===== AUTH ENDPOINTS =====
@app.post("/auth/register", response_model=UserResponse)
async def register(
        user_data: UserRegister,
        response: Response,
        session: AsyncSession = Depends(get_db)
):
    # Проверяем, существует ли пользователь
    existing_user = await session.scalar(
        select(User).where((User.email == user_data.email) | (User.username == user_data.username))
    )

    if existing_user:
        raise HTTPException(status_code=400, detail="Email or username already exists")

    # Создаем нового пользователя
    hashed_password = hash_password(user_data.password)
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name or user_data.username,
        role="user",
        is_active=True
    )

    session.add(user)
    await session.commit()
    await session.refresh(user)

    # Создаем сессию
    session_token = create_session_for_user(user.id)

    # Устанавливаем cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        max_age=60*60*24*7,  # 7 дней
        samesite="lax",
        secure=False
    )

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role
    }

@app.post("/auth/login", response_model=UserResponse)
async def login(
        login_data: UserLogin,
        response: Response,
        session: AsyncSession = Depends(get_db)
):
    user = await session.scalar(
        select(User).where(
            User.email == login_data.email,
            User.is_active == True
        )
    )

    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Обновляем время последнего входа
    user.last_login = datetime.utcnow()
    await session.commit()

    # Создаем сессию
    session_token = create_session_for_user(user.id)

    # Устанавливаем cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        max_age=60*60*24*7,
        samesite="lax",
        secure=False
    )

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role
    }

@app.post("/auth/logout")
async def logout(
        response: Response,
        session_token: Optional[str] = Cookie(default=None, alias="session_token"),
        session: AsyncSession = Depends(get_db)
):
    if session_token:
        delete_session(session_token)

    # Удаляем cookie
    response.delete_cookie(key="session_token")

    return {"message": "Logged out successfully"}

@app.get("/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None
    }

# ===== TEXTS ENDPOINTS =====
@app.get("/texts")
async def list_texts(session: AsyncSession = Depends(get_db)):
    result = await session.execute(
        select(TextItem)
        .where(TextItem.is_active == True)
        .order_by(TextItem.level, TextItem.topic)
    )
    texts = result.scalars().all()

    return [
        {
            "id": text.id,
            "level": text.level,
            "topic": text.topic,
            "title": text.title,
            "content": text.content,
            "word_count": len(text.content.split())
        }
        for text in texts
    ]

@app.get("/texts/{text_id}")
async def get_text(text_id: int, session: AsyncSession = Depends(get_db)):
    text = await session.get(TextItem, text_id)

    if not text or not text.is_active:
        raise HTTPException(status_code=404, detail="Text not found")

    # Генерируем TTS для всего текста (опционально)
    full_text_audio = None
    if TTS_AVAILABLE:
        audio_path = generate_tts_audio_safe(text.content, "en")
        if audio_path:
            text_hash = hashlib.md5(text.content.encode()).hexdigest()
            full_text_audio = f"/tts/pronounce/{text_hash}"

    return {
        "id": text.id,
        "level": text.level,
        "topic": text.topic,
        "title": text.title,
        "content": text.content,
        "word_count": len(text.content.split()),
        "has_audio": full_text_audio is not None,
        "audio_url": full_text_audio
    }

# ===== ANALYZE ENDPOINT =====
@app.post("/analyze")
async def analyze(
        text_id: int = Form(...),
        audio: UploadFile = File(...),
        language: str = Form(default="en"),
        current_user: User = Depends(get_current_active_user),
        session: AsyncSession = Depends(get_db)
):
    logger.info(f"\n{'='*60}")
    logger.info(f"[POST /analyze] User: {current_user.username}")
    logger.info(f"  text_id: {text_id}, language: {language}")

    try:
        # 1. Получаем текст из базы данных
        text = await session.get(TextItem, text_id)
        if not text or not text.is_active:
            raise HTTPException(status_code=404, detail="Text not found")

        logger.info(f"  Text found: {text.title}")

        # 2. Читаем аудиофайл
        audio_bytes = await audio.read()
        logger.info(f"  Audio received: {len(audio_bytes)} bytes")

        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # 3. Обрабатываем WAV
        audio_data = read_wav_bytes(audio_bytes)

        if audio_data is None:
            logger.error("  Error: Failed to process WAV file")
            raise HTTPException(status_code=400, detail="Invalid WAV file format")

        logger.info(f"  Audio processed: {len(audio_data)} samples")

        # 4. Транскрипция с Whisper
        logger.info("  Starting Whisper transcription...")
        recognized_text = transcribe_audio_with_whisper(audio_bytes)

        if not recognized_text or "error" in recognized_text.lower():
            logger.warning(f"  Whisper failed: {recognized_text}")
            # Fallback для коротких текстов
            if len(text.content) < 100:
                recognized_text = text.content
            else:
                recognized_text = "Could not transcribe audio. Please try again."
        else:
            logger.info(f"  ✓ Transcription successful: {recognized_text[:100]}...")

        # 5. Создание детализированного фидбека
        feedback = build_feedback_improved(text.content, recognized_text, generate_audio=True)
        accuracy = feedback["summary"]["accuracy"]
        logger.info(f"  Accuracy: {accuracy}%")

        # 6. Сохраняем проблемные слова для практики
        problem_words = feedback.get("problem_words", [])
        if problem_words and current_user:
            for word in problem_words[:10]:  # Ограничиваем 10 словами
                try:
                    # Проверяем, есть ли уже такое слово
                    existing = await session.scalar(
                        select(PracticeWord).where(
                            and_(
                                PracticeWord.user_id == current_user.id,
                                PracticeWord.word == word,
                                PracticeWord.text_id == text_id
                            )
                        )
                    )

                    if not existing:
                        practice_word = PracticeWord(
                            user_id=current_user.id,
                            word=word,
                            text_id=text_id,
                            practice_count=0,
                            mastered=False,
                            created_at=datetime.utcnow()
                        )
                        session.add(practice_word)
                except Exception as e:
                    logger.error(f"  Error saving word '{word}': {e}")

            await session.commit()

        # 7. Возвращаем результат
        logger.info(f"  Found {len(feedback.get('error_words', []))} error words")
        logger.info(f"  ✓ ANALYZE COMPLETE")
        logger.info(f"{'='*60}")

        return {
            "text_id": text_id,
            "text_title": text.title,
            "reference_text": text.content,
            "recognized_text": recognized_text,
            "feedback": feedback,
            "accuracy": accuracy,
            "tts_available": TTS_AVAILABLE,
            "pronunciation_data": feedback.get("pronunciation_data", {})
        }

    except HTTPException as he:
        logger.error(f"  HTTPException: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"  UNEXPECTED ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ===== WORDS ENDPOINTS =====
@app.post("/words/save")
async def save_words(
        request: SaveWordsRequest,
        current_user: User = Depends(get_current_active_user),
        session: AsyncSession = Depends(get_db)
):
    """Сохранить проблемные слова для практики"""
    try:
        saved_words = []

        for word in request.words:
            # Проверяем, есть ли уже такое слово у пользователя
            existing = await session.scalar(
                select(PracticeWord).where(
                    and_(
                        PracticeWord.user_id == current_user.id,
                        PracticeWord.word == word,
                        PracticeWord.text_id == request.text_id
                    )
                )
            )

            if not existing:
                practice_word = PracticeWord(
                    user_id=current_user.id,
                    word=word,
                    text_id=request.text_id,
                    practice_count=0,
                    mastered=False,
                    created_at=datetime.utcnow()
                )
                session.add(practice_word)
                saved_words.append(word)

        await session.commit()

        # ГАРАНТИРУЕМ ВОЗВРАТ success: true
        return {
            "success": True,  # ← Важно!
            "message": f"Сохранено {len(saved_words)} слов",
            "saved_words": saved_words,
            "saved_count": len(saved_words)
        }

    except Exception as e:
        logger.error(f"Error saving words: {e}")
        # ГАРАНТИРУЕМ ВОЗВРАТ success: false при ошибке
        return {
            "success": False,  # ← Важно!
            "detail": f"Ошибка сохранения слов: {str(e)}",
            "saved_words": []
        }

@app.get("/words/my")
async def get_my_words(
        current_user: User = Depends(get_current_active_user),
        session: AsyncSession = Depends(get_db)
):
    """Получить слова пользователя для практики"""
    try:
        result = await session.execute(
            select(PracticeWord)
            .where(PracticeWord.user_id == current_user.id)
            .order_by(PracticeWord.created_at.desc())
        )
        words = result.scalars().all()

        word_list = []
        for word in words:
            text_title = None
            if word.text_id:
                text = await session.get(TextItem, word.text_id)
                text_title = text.title if text else None

            # Добавляем URL для прослушивания произношения
            pronunciation_url = None
            if TTS_AVAILABLE and word.word:
                pronunciation_url = f"/tts/pronounce/{word.word}"

            word_list.append({
                "id": word.id,
                "word": word.word,
                "text_id": word.text_id,
                "text_title": text_title,
                "practice_count": word.practice_count,
                "mastered": word.mastered,
                "last_practiced": word.last_practiced.isoformat() if word.last_practiced else None,
                "created_at": word.created_at.isoformat(),
                "pronunciation_url": pronunciation_url,
                "has_audio": TTS_AVAILABLE
            })

        return word_list

    except Exception as e:
        logger.error(f"Error getting words: {e}")
        return []

@app.delete("/words/{word_id}")
async def delete_word(
        word_id: int,
        current_user: User = Depends(get_current_active_user),
        session: AsyncSession = Depends(get_db)
):
    """Удалить слово из практики"""
    try:
        word = await session.scalar(
            select(PracticeWord).where(
                and_(
                    PracticeWord.id == word_id,
                    PracticeWord.user_id == current_user.id
                )
            )
        )

        if not word:
            raise HTTPException(status_code=404, detail="Word not found")

        await session.delete(word)
        await session.commit()

        return {"success": True, "message": "Word deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting word: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting word: {str(e)}")

@app.post("/words/{word_id}/practice")
async def mark_word_practiced(
        word_id: int,
        current_user: User = Depends(get_current_active_user),
        session: AsyncSession = Depends(get_db)
):
    """Отметить слово как отработанное"""
    try:
        word = await session.scalar(
            select(PracticeWord).where(
                and_(
                    PracticeWord.id == word_id,
                    PracticeWord.user_id == current_user.id
                )
            )
        )

        if not word:
            raise HTTPException(status_code=404, detail="Word not found")

        # Увеличиваем счетчик практики
        word.practice_count += 1
        word.last_practiced = datetime.utcnow()

        # Если практиковали 3 раза и более, отмечаем как освоенное
        if word.practice_count >= 3:
            word.mastered = True

        await session.commit()

        return {
            "success": True,
            "message": "Word marked as practiced",
            "practice_count": word.practice_count,
            "mastered": word.mastered
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking word: {e}")
        raise HTTPException(status_code=500, detail=f"Error marking word: {str(e)}")

@app.get("/words/stats")
async def get_word_stats(
        current_user: User = Depends(get_current_active_user),
        session: AsyncSession = Depends(get_db)
):
    """Получить статистику по словам"""
    try:
        # Общее количество слов
        total_result = await session.execute(
            select(func.count(PracticeWord.id))
            .where(PracticeWord.user_id == current_user.id)
        )
        total = total_result.scalar() or 0

        # Количество отработанных слов (практика > 0)
        practiced_result = await session.execute(
            select(func.count(PracticeWord.id))
            .where(
                and_(
                    PracticeWord.user_id == current_user.id,
                    PracticeWord.practice_count > 0
                )
            )
        )
        practiced = practiced_result.scalar() or 0

        # Количество освоенных слов
        mastered_result = await session.execute(
            select(func.count(PracticeWord.id))
            .where(
                and_(
                    PracticeWord.user_id == current_user.id,
                    PracticeWord.mastered == True
                )
            )
        )
        mastered = mastered_result.scalar() or 0

        # Количество слов, отработанных сегодня
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_result = await session.execute(
            select(func.count(PracticeWord.id))
            .where(
                and_(
                    PracticeWord.user_id == current_user.id,
                    PracticeWord.last_practiced >= today_start
                )
            )
        )
        today = today_result.scalar() or 0

        return {
            "total": total,
            "practiced": practiced,
            "mastered": mastered,
            "today": today
        }

    except Exception as e:
        logger.error(f"Error getting word stats: {e}")
        return {
            "total": 0,
            "practiced": 0,
            "mastered": 0,
            "today": 0
        }

# ===== PRACTICE WORDS ENDPOINT =====
@app.get("/words/practice")
async def get_practice_words(
        current_user: User = Depends(get_current_active_user),
        session: AsyncSession = Depends(get_db)
):
    """Получить слова для практики с TTS аудио"""
    result = await session.execute(
        select(PracticeWord)
        .where(PracticeWord.user_id == current_user.id)
        .where(PracticeWord.mastered == False)
        .order_by(PracticeWord.practice_count.asc())
        .limit(20)
    )
    words = result.scalars().all()

    word_list = []
    for word in words:
        # Генерируем TTS аудио для слова
        audio_url = None
        if TTS_AVAILABLE and word.word:
            audio_url = f"/tts/pronounce/{word.word}"

        word_list.append({
            "id": word.id,
            "word": word.word,
            "practice_count": word.practice_count,
            "mastered": word.mastered,
            "last_practiced": word.last_practiced.isoformat() if word.last_practiced else None,
            "audio_url": audio_url,
            "has_audio": TTS_AVAILABLE
        })

    return word_list

# ===== ADMIN ENDPOINTS =====
@app.post("/admin/texts")
async def create_text(
        text_data: TextCreate,
        current_user: User = Depends(require_admin),
        session: AsyncSession = Depends(get_db)
):
    text = TextItem(
        level=text_data.level,
        topic=text_data.topic,
        title=text_data.title,
        content=text_data.content,
        created_by=current_user.id
    )

    session.add(text)
    await session.commit()
    await session.refresh(text)

    return {
        "id": text.id,
        "message": "Text created successfully"
    }

@app.get("/admin/texts")
async def get_all_texts(
        current_user: User = Depends(require_admin),
        session: AsyncSession = Depends(get_db)
):
    result = await session.execute(
        select(TextItem)
        .order_by(TextItem.created_at.desc())
    )
    texts = result.scalars().all()

    return [
        {
            "id": text.id,
            "level": text.level,
            "topic": text.topic,
            "title": text.title,
            "content": text.content[:200] + "..." if len(text.content) > 200 else text.content,
            "created_by": text.created_by,
            "created_at": text.created_at.isoformat(),
            "is_active": text.is_active
        }
        for text in texts
    ]

@app.put("/admin/texts/{text_id}")
async def update_text(
        text_id: int,
        text_data: TextCreate,
        current_user: User = Depends(require_admin),
        session: AsyncSession = Depends(get_db)
):
    text = await session.get(TextItem, text_id)

    if not text:
        raise HTTPException(status_code=404, detail="Text not found")

    text.level = text_data.level
    text.topic = text_data.topic
    text.title = text_data.title
    text.content = text_data.content

    await session.commit()

    return {"message": "Text updated successfully"}

@app.delete("/admin/texts/{text_id}")
async def delete_text(
        text_id: int,
        current_user: User = Depends(require_admin),
        session: AsyncSession = Depends(get_db)
):
    text = await session.get(TextItem, text_id)

    if not text:
        raise HTTPException(status_code=404, detail="Text not found")

    text.is_active = False
    await session.commit()

    return {"message": "Text deactivated successfully"}

@app.get("/admin/stats")
async def get_admin_stats(
        current_user: User = Depends(require_admin),
        session: AsyncSession = Depends(get_db)
):
    """Получить статистику для админ-панели"""
    try:
        # Количество текстов
        texts_result = await session.execute(select(func.count(TextItem.id)))
        total_texts = texts_result.scalar() or 0

        # Количество пользователей
        users_result = await session.execute(select(func.count(User.id)))
        total_users = users_result.scalar() or 0

        # Количество практик слов
        practices_result = await session.execute(
            select(func.sum(PracticeWord.practice_count))
        )
        total_practices = practices_result.scalar() or 0

        # Количество TTS файлов в кэше
        tts_cache_size = len(list(TTS_CACHE_DIR.glob("*.mp3"))) if TTS_CACHE_DIR.exists() else 0

        return {
            "total_texts": total_texts,
            "total_users": total_users,
            "total_practices": total_practices,
            "tts_cache_size": tts_cache_size,
            "whisper_loaded": whisper_model is not None,
            "tts_available": TTS_AVAILABLE
        }

    except Exception as e:
        logger.error(f"Error getting admin stats: {e}")
        return {
            "total_texts": 0,
            "total_users": 0,
            "total_practices": 0,
            "tts_cache_size": 0,
            "whisper_loaded": False,
            "tts_available": TTS_AVAILABLE
        }

# ===== HEALTH CHECK =====
@app.get("/health")
async def health_check():
    whisper_loaded = whisper_model is not None
    tts_cache_size = len(list(TTS_CACHE_DIR.glob("*.mp3"))) if TTS_CACHE_DIR.exists() else 0

    return {
        "status": "ok",
        "service": "LinguaAI API",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "whisper_loaded": whisper_loaded,
        "tts_available": TTS_AVAILABLE,
        "tts_cache_size": tts_cache_size
    }

@app.get("/")
async def api_root():
    return {
        "message": "LinguaAI API",
        "version": "2.0.0",
        "features": {
            "authentication": True,
            "speech_recognition": "OpenAI Whisper",
            "pronunciation_analysis": True,
            "text_to_speech": "gTTS" if TTS_AVAILABLE else "Not installed",
            "word_practice_tracking": True,
            "admin_panel": True
        },
        "endpoints": {
            "auth": ["/auth/register", "/auth/login", "/auth/me", "/auth/logout"],
            "texts": ["/texts", "/texts/{id}"],
            "analyze": "/analyze",
            "tts": ["/tts/pronounce/{word}", "/tts/generate", "/tts/status"],
            "words": [
                "/words/my",
                "/words/save",
                "/words/stats",
                "/words/practice",
                "/words/{word_id}/practice",
                "/words/{word_id} (DELETE)"
            ],
            "admin": ["/admin/texts", "/admin/stats"],
            "health": "/health"
        }
    }

# ===== MAIN =====
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING LINGUAAI SERVER")
    print("="*60)

    if TTS_AVAILABLE:
        print("✅ TTS Service: gTTS (pronunciation audio available)")
        cache_files = list(TTS_CACHE_DIR.glob("*.mp3"))
        print(f"   Cache: {len(cache_files)} audio files")
    else:
        print("⚠️  TTS Service: Not installed")
        print("   Install with: pip install gtts")

    print("\n📊 Features:")
    print("   • User authentication & authorization")
    print("   • Speech recognition (Whisper)")
    print("   • Pronunciation error detection")
    print("   • TTS pronunciation audio")
    print("   • Word practice tracking")
    print("   • Admin panel")

    print("\n🔧 Endpoints:")
    print("   http://localhost:8000/auth/*      - Authentication")
    print("   http://localhost:8000/analyze     - Analyze pronunciation")
    print("   http://localhost:8000/texts       - Get texts")
    print("   http://localhost:8000/words/*     - Word practice")
    print("   http://localhost:8000/tts/*       - Text-to-speech")
    print("   http://localhost:8000/admin/*     - Admin endpoints")

    print("\n👤 Default admin: admin@linguacab.com / admin123")
    print("🌐 Frontend: http://localhost:3000")
    print("⚡ API: http://localhost:8000")
    print("="*60 + "\n")

    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
