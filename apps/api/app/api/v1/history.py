"""API endpoints for historical prediction retrieval, filtering, and inspection."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_analytics_service, get_db, get_prediction_service
from app.schemas.analytics import PaginatedPredictionsResponse
from app.schemas.prediction import PredictionDetailSchema
from app.services.analytics_service import AnalyticsService
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/history", tags=["History"])


@router.get(
    "/predictions",
    summary="List Historical Predictions",
    response_model=PaginatedPredictionsResponse,
    status_code=status.HTTP_200_OK,
    response_description="Filterable, paginated prediction records with face bounding boxes and confidence",
)
async def list_prediction_history(
    session_id: uuid.UUID | None = Query(default=None, description="Filter by session UUID"),
    emotion: str | None = Query(
        default=None, description="Filter by predicted emotion class (e.g. happy)"
    ),
    is_uncertain: bool | None = Query(
        default=None, description="Filter by uncertainty flag"
    ),
    min_confidence: float | None = Query(
        default=None, ge=0.0, le=1.0, description="Minimum confidence score"
    ),
    max_confidence: float | None = Query(
        default=None, ge=0.0, le=1.0, description="Maximum confidence score"
    ),
    start_date: datetime | None = Query(
        default=None, description="Filter predictions on/after ISO timestamp"
    ),
    end_date: datetime | None = Query(
        default=None, description="Filter predictions on/before ISO timestamp"
    ),
    model_version: str | None = Query(
        default=None, description="Filter by model version string"
    ),
    limit: int = Query(default=50, ge=1, le=200, description="Page limit"),
    offset: int = Query(default=0, ge=0, description="Page offset"),
    sort_by: str = Query(
        default="created_at",
        pattern="^(created_at|confidence)$",
        description="Sort field: created_at or confidence",
    ),
    order: str = Query(
        default="desc", pattern="^(asc|desc)$", description="Sort order: asc or desc"
    ),
    db: AsyncSession = Depends(get_db),
    service: AnalyticsService = Depends(get_analytics_service),
) -> PaginatedPredictionsResponse:
    """Retrieve historical prediction events with server-side filtering, sorting, and pagination."""
    return await service.list_prediction_history(
        db=db,
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


@router.get(
    "/predictions/{prediction_id}",
    summary="Inspect Prediction Record",
    response_model=PredictionDetailSchema,
    status_code=status.HTTP_200_OK,
    response_description="Detailed prediction record with detected faces, probabilities, and model traceability",
)
async def inspect_prediction(
    prediction_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    pred_service: PredictionService = Depends(get_prediction_service),
) -> PredictionDetailSchema:
    """Retrieve full details of a specific prediction record for inspection and explainability analysis."""
    return await pred_service.get_prediction_by_id(db=db, prediction_id=prediction_id)
