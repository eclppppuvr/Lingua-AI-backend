# auth.py - сессии в Redis вместо памяти процесса
import bcrypt
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import HTTPException, Cookie, Response, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, Boolean, DateTime, Text
from pydantic import BaseModel
import json
import redis

from database import Base, get_db

# ===== PYDANTIC MODELS =====
class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str

# ===== DATABASE MODELS =====
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    role: Mapped[str] = mapped_column(String(20), default="user")  # user, admin
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

# ===== REDIS SESSION STORAGE =====
class RedisSessionStorage:
    def __init__(self):
        # Подключаемся к Redis
        self.redis = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True  # Автоматически декодируем в строки
        )
        
        # Проверяем соединение
        try:
            self.redis.ping()
            print("✓ Redis connected successfully")
        except redis.ConnectionError:
            print("✗ Redis connection failed. Using in-memory fallback.")
            self.redis = None
            self.fallback_storage = {}

    def create_session(self, user_id: int) -> str:
        """Создает новую сессию в Redis"""
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            'user_id': user_id,
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(days=7)).isoformat()
        }
        
        if self.redis:
            # Сохраняем в Redis на 7 дней
            self.redis.setex(
                f"session:{session_id}",
                60 * 60 * 24 * 7,  # 7 дней в секундах
                json.dumps(session_data)
            )
        else:
            # Фолбэк в память (если Redis не доступен)
            self.fallback_storage[session_id] = session_data
            
        return session_id

    def get_user_id(self, session_id: str) -> Optional[int]:
        """Получает user_id по session_id из Redis"""
        if not session_id:
            return None
        
        try:
            if self.redis:
                # Получаем из Redis
                session_json = self.redis.get(f"session:{session_id}")
                if not session_json:
                    return None
                    
                session_data = json.loads(session_json)
            else:
                # Фолбэк из памяти
                if session_id not in self.fallback_storage:
                    return None
                session_data = self.fallback_storage[session_id]
            
            # Проверяем не истекла ли сессия
            expires_at = datetime.fromisoformat(session_data['expires_at'])
            if expires_at < datetime.utcnow():
                self.delete_session(session_id)
                return None
            
            # Обновляем время жизни (только в Redis)
            if self.redis:
                self.redis.expire(f"session:{session_id}", 60 * 60 * 24 * 7)
            
            return session_data['user_id']
            
        except Exception as e:
            print(f"Error getting session: {e}")
            return None

    def delete_session(self, session_id: str):
        """Удаляет сессию"""
        try:
            if self.redis:
                self.redis.delete(f"session:{session_id}")
            elif session_id in self.fallback_storage:
                del self.fallback_storage[session_id]
        except Exception as e:
            print(f"Error deleting session: {e}")

# Глобальное хранилище сессий (теперь в Redis)
session_storage = RedisSessionStorage()

# ===== PASSWORD UTILITIES =====
def hash_password(password: str) -> str:
    """Хеширует пароль"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверяет пароль"""
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    except Exception:
        return False

# ===== AUTH DEPENDENCIES =====
async def get_current_user(
        session_id: Optional[str] = Cookie(default=None, alias="session_token"),
        session: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Получает текущего пользователя по session_id из cookie"""
    if not session_id:
        return None

    # Получаем user_id из хранилища сессий (теперь из Redis)
    user_id = session_storage.get_user_id(session_id)
    if not user_id:
        return None

    # Находим пользователя в базе
    result = await session.execute(
        select(User).where(
            User.id == user_id,
            User.is_active == True
        )
    )
    user = result.scalar_one_or_none()

    return user

async def get_current_active_user(
        current_user: Optional[User] = Depends(get_current_user)
) -> User:
    """Проверяет что пользователь аутентифицирован"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return current_user

def require_admin(current_user: User = Depends(get_current_active_user)) -> User:
    """Проверяет что пользователь администратор"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user

# ===== SESSION MANAGEMENT =====
def create_session_for_user(user_id: int) -> str:
    """Создает сессию для пользователя"""
    return session_storage.create_session(user_id)

def delete_session(session_id: str):
    """Удаляет сессию"""
    session_storage.delete_session(session_id)

# Функция для очистки старых сессий (опционально)
def cleanup_old_sessions():
    """Очищает старые сессии (можно запускать по cron)"""
    # Redis сам удаляет истекшие ключи
    pass
