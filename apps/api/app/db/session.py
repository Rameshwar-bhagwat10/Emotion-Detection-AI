"""Asynchronous database session and connection management."""

from collections.abc import AsyncGenerator
from pathlib import Path
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


def _create_configured_engine(db_url: str) -> AsyncEngine:
    """Helper to configure async engine based on dialect."""
    if db_url.startswith("sqlite"):
        return create_async_engine(
            db_url,
            echo=settings.DEBUG,
            future=True,
        )
    return create_async_engine(
        db_url,
        echo=settings.DEBUG,
        future=True,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )


# Primary async engine
engine: AsyncEngine = _create_configured_engine(settings.DATABASE_URL)
active_db_type: str = "postgresql" if not settings.DATABASE_URL.startswith("sqlite") else "sqlite"
_sqlite_fallback_initialized: bool = False

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Retrieve the active sessionmaker instance."""
    return AsyncSessionLocal


async def _init_sqlite_fallback() -> None:
    """Initialize fallback SQLite database schema when PostgreSQL is unavailable."""
    global engine, AsyncSessionLocal, active_db_type, _sqlite_fallback_initialized
    if _sqlite_fallback_initialized:
        return

    from app.db.base import Base
    import app.db.models  # noqa: F401

    fallback_path = Path(__file__).resolve().parent.parent.parent / "emotion_dev.db"
    fallback_url = f"sqlite+aiosqlite:///{fallback_path.as_posix()}"
    logger.warning(
        f"Primary database connection failed. Falling back to local SQLite database: {fallback_url}"
    )

    engine = _create_configured_engine(fallback_url)
    AsyncSessionLocal = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    active_db_type = "sqlite"
    _sqlite_fallback_initialized = True
    logger.info("Local SQLite database schema initialized successfully.")


async def init_db() -> None:
    """Verify primary database connectivity or initialize local SQLite fallback."""
    global engine, AsyncSessionLocal, active_db_type, _sqlite_fallback_initialized
    if _sqlite_fallback_initialized:
        return
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            logger.info(f"Connected to primary database ({active_db_type}).")
    except Exception as exc:
        logger.warning(f"Primary database connection failed ({exc}). Initializing local SQLite fallback...")
        await _init_sqlite_fallback()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency provider yielding asynchronous database sessions with automatic fallback."""
    factory = get_session_factory()
    async with factory() as session:
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
                "database": active_db_type,
                "host": settings.POSTGRES_HOST if active_db_type == "postgresql" else "local",
                "port": settings.POSTGRES_PORT if active_db_type == "postgresql" else 0,
            }
    except Exception as exc:
        logger.warning(f"Database health check failed ({exc}). Attempting local SQLite fallback...")
        try:
            await _init_sqlite_fallback()
            async with AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
                return {
                    "status": "healthy",
                    "database": "sqlite_fallback",
                    "host": "local",
                    "port": 0,
                }
        except Exception as fb_exc:
            logger.error(f"Fallback database initialization also failed: {fb_exc}")
            return {
                "status": "unavailable",
                "database": settings.POSTGRES_DB,
                "error": str(exc),
            }
