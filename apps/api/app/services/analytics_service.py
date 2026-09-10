"""Service layer for aggregating and orchestrating facial expression analytics and history."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ResourceNotFoundError
from app.db.repositories.analytics import AnalyticsRepository
from app.schemas.analytics import (
    ConfidenceAnalyticsSchema,
    DailyTrendItem,
    ExpressionDistributionSchema,
    GlobalAnalyticsResponse,
    HistoricalPredictionItem,
    PaginatedPredictionsResponse,
    SessionAnalyticsListResponse,
    SessionAnalyticsResponse,
    SessionSummaryItem,
    TimelineAnalyticsResponse,
)

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Coordinates business logic for global overview, session analytics, timelines, and history."""

    def __init__(self, analytics_repo: AnalyticsRepository | None = None) -> None:
        self.repo = analytics_repo or AnalyticsRepository()

    async def get_global_overview(
        self,
        db: AsyncSession,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        model_version: str | None = None,
    ) -> GlobalAnalyticsResponse:
        """Fetch and structure system-wide prediction and session analytics."""
        raw = await self.repo.get_global_overview(
            session=db,
            start_date=start_date,
            end_date=end_date,
            model_version=model_version,
        )

        return GlobalAnalyticsResponse(
            total_sessions=raw["total_sessions"],
            total_predictions=raw["total_predictions"],
            total_faces=raw["total_faces"],
            total_duration_seconds=raw["total_duration_seconds"],
            dominant_expression=raw["dominant_expression"],
            average_confidence=raw["average_confidence"],
            expression_distribution=ExpressionDistributionSchema(
                items=raw["expression_distribution"]["items"],
                dominant_emotion=raw["expression_distribution"]["dominant_emotion"],
                total_predictions=raw["expression_distribution"]["total_predictions"],
            ),
            confidence_analytics=ConfidenceAnalyticsSchema(
                average_confidence=raw["confidence_analytics"]["average_confidence"],
                min_confidence=raw["confidence_analytics"]["min_confidence"],
                max_confidence=raw["confidence_analytics"]["max_confidence"],
                distribution=raw["confidence_analytics"]["distribution"],
                class_confidences=raw["confidence_analytics"]["class_confidences"],
                low_confidence_count=raw["confidence_analytics"]["low_confidence_count"],
                high_confidence_count=raw["confidence_analytics"]["high_confidence_count"],
            ),
            model_version_distribution=raw["model_version_distribution"],
            recent_trends=[DailyTrendItem(**item) for item in raw["recent_trends"]],
        )

    async def get_session_analytics(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
    ) -> SessionAnalyticsResponse:
        """Fetch and structure analytics for a single target session."""
        raw = await self.repo.get_session_analytics(session=db, session_id=session_id)
        if raw is None:
            raise ResourceNotFoundError(
                message=f"Analysis session '{session_id}' not found",
                details={"session_id": str(session_id)},
            )

        return SessionAnalyticsResponse(
            session_id=raw["session_id"],
            name=raw["name"],
            status=raw["status"],
            started_at=raw["started_at"],
            ended_at=raw["ended_at"],
            duration_seconds=raw["duration_seconds"],
            total_predictions=raw["total_predictions"],
            total_faces=raw["total_faces"],
            prediction_rate_per_minute=raw["prediction_rate_per_minute"],
            dominant_expression=raw["dominant_expression"],
            expression_distribution=ExpressionDistributionSchema(
                items=raw["expression_distribution"]["items"],
                dominant_emotion=raw["expression_distribution"]["dominant_emotion"],
                total_predictions=raw["expression_distribution"]["total_predictions"],
            ),
            confidence_analytics=ConfidenceAnalyticsSchema(
                average_confidence=raw["confidence_analytics"]["average_confidence"],
                min_confidence=raw["confidence_analytics"]["min_confidence"],
                max_confidence=raw["confidence_analytics"]["max_confidence"],
                distribution=raw["confidence_analytics"]["distribution"],
                class_confidences=raw["confidence_analytics"]["class_confidences"],
                low_confidence_count=raw["confidence_analytics"]["low_confidence_count"],
                high_confidence_count=raw["confidence_analytics"]["high_confidence_count"],
            ),
            model_versions=raw["model_versions"],
        )

    async def get_session_timeline(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        bucket_seconds: int | None = None,
    ) -> TimelineAnalyticsResponse:
        """Fetch time-bucketed emotion events for temporal session visualization."""
        raw = await self.repo.get_session_timeline(
            session=db,
            session_id=session_id,
            bucket_seconds=bucket_seconds,
        )
        if raw is None:
            raise ResourceNotFoundError(
                message=f"Analysis session '{session_id}' not found",
                details={"session_id": str(session_id)},
            )

        return TimelineAnalyticsResponse(**raw)

    async def list_prediction_history(
        self,
        db: AsyncSession,
        session_id: uuid.UUID | None = None,
        emotion: str | None = None,
        is_uncertain: bool | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        model_version: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "created_at",
        order: str = "desc",
    ) -> PaginatedPredictionsResponse:
        """Fetch historical face predictions with filters and server-side pagination."""
        items, total_count = await self.repo.list_prediction_history(
            session=db,
            session_id=session_id,
            emotion=emotion,
            is_uncertain=is_uncertain,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            start_date=start_date,
            end_date=end_date,
            model_version=model_version,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            order=order,
        )

        history_items = [HistoricalPredictionItem(**item) for item in items]
        return PaginatedPredictionsResponse(
            items=history_items,
            total=total_count,
            limit=limit,
            offset=offset,
        )

    async def list_sessions_with_analytics(
        self,
        db: AsyncSession,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> SessionAnalyticsListResponse:
        """List sessions enriched with computed summary analytics."""
        items, total_count = await self.repo.list_sessions_with_analytics(
            session=db,
            limit=limit,
            offset=offset,
            status=status,
            start_date=start_date,
            end_date=end_date,
        )

        session_summaries = [SessionSummaryItem(**item) for item in items]
        return SessionAnalyticsListResponse(
            sessions=session_summaries,
            total=total_count,
        )
