"""Central application settings and environment configuration."""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # General App Config
    APP_ENV: str = Field(default="development", description="Application environment")
    APP_NAME: str = Field(default="emotion-detection-api", description="Service name")
    APP_VERSION: str = Field(default="1.0.0", description="API Version")
    DEBUG: bool = Field(default=True, description="Debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    SECRET_KEY: str = Field(
        default="dev-secret-key-change-in-production-min-32-chars-long!",
        description="JWT and encryption secret key",
    )

    # CORS
    CORS_ORIGINS: list[str] | str = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        """Parse CORS origins from JSON list or comma-separated string."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                return [origin.strip() for origin in v.split(",") if origin.strip()]
        elif isinstance(v, list):
            return [str(x) for x in v]
        return ["http://localhost:3000", "http://127.0.0.1:3000"]

    # API Server
    API_HOST: str = Field(default="0.0.0.0", description="Host to bind API server")
    API_PORT: int = Field(default=8000, description="Port for API server")
    API_BASE_URL: str = Field(
        default="http://localhost:3000/api/v1", description="Base URL of API service"
    )

    # Supabase Integration
    SUPABASE_URL: str = Field(default="", description="Supabase project URL")
    SUPABASE_ANON_KEY: str = Field(default="", description="Supabase anonymous public key")
    SUPABASE_PUBLISHABLE_KEY: str = Field(default="", description="Supabase publishable key")

    # PostgreSQL Database
    POSTGRES_DB: str = Field(default="emotion_detection", description="Postgres DB name")
    POSTGRES_USER: str = Field(default="emotion_user", description="Postgres user")
    POSTGRES_PASSWORD: str = Field(default="change_me", description="Postgres password")
    POSTGRES_HOST: str = Field(default="localhost", description="Postgres host")
    POSTGRES_PORT: int = Field(default=5432, description="Postgres port")
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://emotion_user:change_me@localhost:5432/emotion_detection",
        description="Async SQLAlchemy database URL",
    )
    DATABASE_URL_SYNC: str = Field(
        default="postgresql://emotion_user:change_me@localhost:5432/emotion_detection",
        description="Sync SQLAlchemy database URL for Alembic",
    )

    # Redis Cache & PubSub
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_DB: int = Field(default=0, description="Redis database index")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")

    # Celery Background Worker
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/0", description="Celery broker URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/1", description="Celery result backend URL"
    )

    # ML & Storage
    MODEL_PATH: str = Field(
        default="artifacts/optimized/champion",
        description="Path to Phase 08 Optimized Champion directory",
    )
    MODEL_VERSION: str = Field(
        default="champion-pruning-30",
        description="Champion model identifier",
    )
    DEVICE: str = Field(
        default="auto", description="Compute device for ML inference (auto, cpu, cuda)"
    )
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.40,
        description="Application confidence threshold for uncertain emotion labeling",
    )
    MAX_IMAGE_SIZE_MB: int = Field(
        default=10,
        description="Maximum allowed image upload payload size in megabytes",
    )
    ALLOWED_IMAGE_EXTENSIONS: list[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".webp"],
        description="Allowed file extensions for image uploads",
    )

    # Real-Time Video & WebSocket Stream
    REALTIME_TARGET_FPS: int = Field(
        default=10,
        description="Target processing rate for real-time video stream (FPS)",
    )
    REALTIME_MAX_FRAME_SIZE_MB: int = Field(
        default=5,
        description="Maximum allowed size for a single real-time frame (MB)",
    )
    REALTIME_QUEUE_MAX_SIZE: int = Field(
        default=2,
        description="Maximum queue buffer size for real-time backpressure control",
    )
    REALTIME_STALE_FRAME_MS: int = Field(
        default=350,
        description="Maximum age in milliseconds before a buffered frame is considered stale and dropped",
    )
    REALTIME_SMOOTHING_ALPHA: float = Field(
        default=0.6,
        description="Exponential moving average alpha for temporal emotion smoothing (0.0 to 1.0)",
    )
    REALTIME_PERSISTENCE_INTERVAL_SEC: float = Field(
        default=1.0,
        description="Minimum interval in seconds between persistent database records for a real-time session",
    )


settings = Settings()
