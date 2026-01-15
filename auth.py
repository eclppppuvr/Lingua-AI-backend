# auth.py - простой вариант авторизации без JWT
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

# ===== SESSION STORAGE (в памяти) =====
class SessionStorage:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def create_session(self, user_id: int) -> str:
        """Создает новую сессию"""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(days=7)
        }
        return session_id

    def get_user_id(self, session_id: str) -> Optional[int]:
        """Получает user_id по session_id"""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # Проверяем не истекла ли сессия
        if session['expires_at'] < datetime.utcnow():
            del self.sessions[session_id]
            return None

        # Обновляем время жизни сессии
        session['expires_at'] = datetime.utcnow() + timedelta(days=7)
        return session['user_id']

    def delete_session(self, session_id: str):
        """Удаляет сессию"""
        if session_id in self.sessions:
            del self.sessions[session_id]

# Глобальное хранилище сессий
session_storage = SessionStorage()

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

    # Получаем user_id из хранилища сессий
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