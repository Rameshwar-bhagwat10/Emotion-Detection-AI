"""Central application settings and environment configuration."""

import json

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
    APP_VERSION: str = Field(default="0.1.0", description="API Version")
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
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # API Server
    API_HOST: str = Field(default="0.0.0.0", description="Host to bind API server")
    API_PORT: int = Field(default=8000, description="Port for API server")

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
    MODEL_PATH: str = Field(default="models/production", description="Path to trained models")
    DEVICE: str = Field(default="auto", description="Compute device for ML inference")


settings = Settings()
