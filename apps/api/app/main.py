"""FastAPI Application Entrypoint for Emotion Detection AI."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.db.session import engine
from app.integrations.redis import close_redis_client

# Initialize structured logging
setup_logging(log_level=settings.LOG_LEVEL)
logger = get_logger("emotion-api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle management for startup and graceful shutdown."""
    logger.info(
        f"Starting {settings.APP_NAME} [v{settings.APP_VERSION}] in {settings.APP_ENV} mode"
    )
    yield
    logger.info(f"Shutting down {settings.APP_NAME}...")
    await close_redis_client()
    await engine.dispose()
    logger.info("Application shutdown complete.")


def create_application() -> FastAPI:
    """Factory creating and configuring the FastAPI application instance."""
    app = FastAPI(
        title="Emotion Detection AI Backend",
        description="FastAPI service for real-time facial expression analysis and emotion detection.",
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register custom exception handlers
    register_exception_handlers(app)

    # Mount API v1 router
    app.include_router(api_v1_router, prefix="/api/v1")

    # Root redirect / convenience health endpoint
    @app.get("/", tags=["Root"])
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "message": "AI-Based Facial Expression Emotion Detection API",
                "version": settings.APP_VERSION,
                "docs": "/docs",
                "health": "/api/v1/health",
            }
        )

    return app


app = create_application()
