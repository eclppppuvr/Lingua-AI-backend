# database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os

# Используем асинхронный SQLite
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite+aiosqlite:///{os.path.join(BASE_DIR, 'linguacab.db')}"

# Создаем асинхронный движок
engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True
)

# Создаем фабрику асинхронных сессий
SessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Единый Base для всех моделей
Base = declarative_base()

# Функция для получения сессии БД
async def get_db():
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()