"""Health check endpoints for service and dependency monitoring."""

from typing import Any

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.db.session import check_db_health
from app.integrations.redis import check_redis_health

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Comprehensive Health Check",
    response_description="Detailed system and dependency health status",
)
async def health_check() -> JSONResponse:
    """Check API server, PostgreSQL database, and Redis connectivity."""
    db_health = await check_db_health()
    redis_health = await check_redis_health()

    # Determine overall system health
    is_fully_healthy = (
        db_health.get("status") == "healthy" and redis_health.get("status") == "healthy"
    )
    overall_status = "ok" if is_fully_healthy else "degraded"

    payload: dict[str, Any] = {
        "status": overall_status,
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "dependencies": {
            "database": db_health,
            "redis": redis_health,
        },
    }

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=payload,
    )


@router.get(
    "/health/live",
    summary="Liveness Probe",
    response_description="Quick liveness indicator for orchestrators",
)
async def liveness_probe() -> dict[str, str]:
    """Lightweight endpoint to confirm the web process is running."""
    return {
        "status": "ok",
        "service": settings.APP_NAME,
    }
