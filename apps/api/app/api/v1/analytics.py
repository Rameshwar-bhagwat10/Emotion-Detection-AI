"""API endpoints for Phase 13 system-level and session-level facial expression analytics."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_analytics_service, get_db
from app.schemas.analytics import (
    ConfidenceAnalyticsSchema,
    GlobalAnalyticsResponse,
    SessionAnalyticsListResponse,
    SessionAnalyticsResponse,
    TimelineAnalyticsResponse,
)
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get(
    "/overview",
    summary="Global Analytics Overview",
    response_model=GlobalAnalyticsResponse,
    status_code=status.HTTP_200_OK,
    response_description="System-wide aggregated facial expression statistics and trends",
)
async def get_global_analytics(
    start_date: datetime | None = Query(
        default=None, description="Optional ISO filter start timestamp"
    ),
    end_date: datetime | None = Query(
        default=None, description="Optional ISO filter end timestamp"
    ),
    model_version: str | None = Query(
        default=None, description="Filter predictions by model version"
    ),
    db: AsyncSession = Depends(get_db),
    service: AnalyticsService = Depends(get_analytics_service),
) -> GlobalAnalyticsResponse:
    """Retrieve system-wide aggregated metrics, emotion distributions, confidence histograms, and trends."""
    return await service.get_global_overview(
        db=db,
        start_date=start_date,
        end_date=end_date,
        model_version=model_version,
    )


@router.get(
    "/sessions",
    summary="List Sessions with Computed Analytics",
    response_model=SessionAnalyticsListResponse,
    status_code=status.HTTP_200_OK,
    response_description="Paginated list of sessions enriched with dominant emotion and avg confidence",
)
async def list_sessions_with_analytics(
    limit: int = Query(default=20, ge=1, le=100, description="Max sessions to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    status_filter: str | None = Query(
        default=None, alias="status", description="Filter by status (active, completed, cancelled)"
    ),
    start_date: datetime | None = Query(
        default=None, description="Filter sessions started on/after"
    ),
    end_date: datetime | None = Query(
        default=None, description="Filter sessions started on/before"
    ),
    db: AsyncSession = Depends(get_db),
    service: AnalyticsService = Depends(get_analytics_service),
) -> SessionAnalyticsListResponse:
    """List historical sessions enriched with prediction counts, duration, and dominant expression."""
    return await service.list_sessions_with_analytics(
        db=db,
        limit=limit,
        offset=offset,
        status=status_filter,
        start_date=start_date,
        end_date=end_date,
    )


@router.get(
    "/sessions/{session_id}",
    summary="Get Session Analytics",
    response_model=SessionAnalyticsResponse,
    status_code=status.HTTP_200_OK,
    response_description="Complete analytics breakdown isolated to the specified session",
)
async def get_session_analytics(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    service: AnalyticsService = Depends(get_analytics_service),
) -> SessionAnalyticsResponse:
    """Retrieve duration, prediction counts, dominant expression, and confidence for a single session."""
    return await service.get_session_analytics(db=db, session_id=session_id)


@router.get(
    "/sessions/{session_id}/timeline",
    summary="Get Session Expression Timeline",
    response_model=TimelineAnalyticsResponse,
    status_code=status.HTTP_200_OK,
    response_description="Time-bucketed emotion predictions for temporal visualization",
)
async def get_session_timeline(
    session_id: uuid.UUID,
    bucket_seconds: int | None = Query(
        default=None, ge=1, le=300, description="Width of each time bucket in seconds"
    ),
    db: AsyncSession = Depends(get_db),
    service: AnalyticsService = Depends(get_analytics_service),
) -> TimelineAnalyticsResponse:
    """Retrieve temporal emotion progression buckets across the session duration."""
    return await service.get_session_timeline(
        db=db,
        session_id=session_id,
        bucket_seconds=bucket_seconds,
    )


@router.get(
    "/sessions/{session_id}/confidence",
    summary="Get Session Confidence Metrics",
    response_model=ConfidenceAnalyticsSchema,
    status_code=status.HTTP_200_OK,
    response_description="Confidence distribution histogram and uncertainty counts for session",
)
async def get_session_confidence(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    service: AnalyticsService = Depends(get_analytics_service),
) -> ConfidenceAnalyticsSchema:
    """Retrieve confidence metrics, 5-bin histogram, and uncertainty stats for a session."""
    analytics = await service.get_session_analytics(db=db, session_id=session_id)
    return analytics.confidence_analytics
