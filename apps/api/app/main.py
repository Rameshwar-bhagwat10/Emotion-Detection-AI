"""FastAPI Application Entrypoint for Emotion Detection AI Backend."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.db.session import engine
from app.integrations.redis import close_redis_client
from ml.inference.config import InferencePipelineConfig
from ml.inference.engine import EmotionInferenceEngine

# Initialize structured logging
setup_logging(log_level=settings.LOG_LEVEL)
logger = get_logger("emotion-api")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware for generating correlation Request IDs and tracking request latency."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.perf_counter()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

        logger.info(
            f"[{request_id}] {request.method} {request.url.path} -> {response.status_code} ({duration_ms:.2f}ms)"
        )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle management: initialize ML engine, warm up, and graceful shutdown."""
    logger.info(
        f"Starting {settings.APP_NAME} [v{settings.APP_VERSION}] in {settings.APP_ENV} mode"
    )

    # 1. Initialize Phase 09 ML Inference Engine
    try:
        pipeline_config = InferencePipelineConfig()
        logger.info(
            f"Loading Phase 09 Champion model: {pipeline_config.model.champion_dir} ({pipeline_config.device.strategy})"
        )
        inference_engine = EmotionInferenceEngine(config=pipeline_config)
        logger.info("Executing Phase 09 model warm-up...")
        inference_engine.warm_up(num_warmup_passes=3)
        app.state.inference_engine = inference_engine
        app.state.is_ready = True
        logger.info("Phase 09 Inference Engine successfully loaded and ready.")
    except Exception as exc:
        logger.error(f"Failed to initialize Phase 09 Inference Engine: {exc}", exc_info=True)
        app.state.inference_engine = None
        app.state.is_ready = False

    yield

    # Teardown
    logger.info(f"Shutting down {settings.APP_NAME}...")
    await close_redis_client()
    await engine.dispose()
    logger.info("Application shutdown complete.")


def create_application() -> FastAPI:
    """Factory creating and configuring the FastAPI application instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        description="Production FastAPI service for real-time facial expression analysis and emotion detection.",
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # Add Request Context Middleware
    app.add_middleware(RequestContextMiddleware)

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

    # Root redirect / metadata endpoint
    @app.get("/", tags=["Root"])
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "message": "AI-Based Facial Expression Emotion Detection API",
                "version": settings.APP_VERSION,
                "docs": "/docs",
                "redoc": "/redoc",
                "health": "/api/v1/health",
                "ready": "/api/v1/health/ready",
            }
        )

    return app


app = create_application()
