"""Health and readiness check endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.db.session import check_db_health
from app.integrations.redis import check_redis_health
from app.integrations.supabase import check_supabase_health
from app.schemas.common import HealthResponse, ReadinessResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Service Health Probe",
    response_model=HealthResponse,
    response_description="Liveness and basic dependency status",
)
async def health_check() -> JSONResponse:
    """Basic health probe returning current status and environment metadata."""
    db_health = await check_db_health()
    supabase_health = await check_supabase_health()
    redis_health = await check_redis_health()

    is_healthy = db_health.get("status") == "healthy" or supabase_health.get("status") == "healthy"
    overall_status = "ok" if is_healthy else "degraded"

    payload: dict[str, Any] = {
        "status": overall_status,
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "dependencies": {
            "database": db_health,
            "supabase": supabase_health,
            "redis": redis_health,
        },
    }

    return JSONResponse(status_code=status.HTTP_200_OK, content=payload)


@router.get(
    "/health/live",
    summary="Container Liveness Probe",
    response_description="Rapid status indicator for orchestrators",
)
async def liveness_probe() -> dict[str, str]:
    """Lightweight endpoint to confirm the API server event loop is active."""
    return {
        "status": "ok",
        "service": settings.APP_NAME,
    }


@router.get(
    "/health/ready",
    summary="Service Readiness Probe",
    response_model=ReadinessResponse,
    response_description="Deep readiness verification across database and ML inference engine",
)
async def readiness_probe(request: Request) -> JSONResponse:
    """Deep readiness probe ensuring model is loaded and database is reachable."""
    db_health = await check_db_health()
    db_ready = db_health.get("status") == "healthy"

    inference_engine = getattr(request.app.state, "inference_engine", None)
    if inference_engine is None:
        try:
            from ml.inference.config import InferencePipelineConfig
            from ml.inference.engine import EmotionInferenceEngine
            pipeline_config = InferencePipelineConfig()
            inference_engine = EmotionInferenceEngine(config=pipeline_config)
            inference_engine.warm_up(num_warmup_passes=1)
            request.app.state.inference_engine = inference_engine
            request.app.state.is_ready = True
            request.app.state.model_error = None
        except Exception as exc:
            request.app.state.model_error = str(exc)

    model_ready = inference_engine is not None and getattr(request.app.state, "is_ready", False)

    is_ready = db_ready and model_ready
    status_str = "ready" if is_ready else "not_ready"
    http_status = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    payload: dict[str, Any] = {
        "status": status_str,
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "model": "ready" if model_ready else "not_ready",
        "database": "ready" if db_ready else "not_ready",
        "details": {
            "model_version": settings.MODEL_VERSION,
            "device": settings.DEVICE,
            "database": db_health,
            "model_error": getattr(request.app.state, "model_error", None),
        },
    }

    return JSONResponse(status_code=http_status, content=payload)
