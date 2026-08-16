"""Asynchronous database session and connection management."""

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Initialize async engine
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency provider yielding asynchronous database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            logger.error(f"Database session error: {exc}")
            raise
        finally:
            await session.close()


async def check_db_health() -> dict[str, Any]:
    """Verify database connectivity via a lightweight ping query.

    Returns:
        Dictionary indicating status, response latency, or error message.
    """
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            return {
                "status": "healthy",
                "database": settings.POSTGRES_DB,
                "host": settings.POSTGRES_HOST,
                "port": settings.POSTGRES_PORT,
            }
    except Exception as exc:
        logger.warning(f"Database health check failed: {str(exc)}")
        return {
            "status": "unavailable",
            "database": settings.POSTGRES_DB,
            "error": "Could not connect to PostgreSQL instance",
        }
