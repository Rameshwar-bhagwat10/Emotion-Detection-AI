"""API endpoints for facial emotion prediction and prediction metadata retrieval."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, get_prediction_service, get_request_id
from app.schemas.prediction import PredictionDetailSchema, PredictionResponseSchema
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post(
    "",
    summary="Predict Facial Emotions from Image",
    response_model=PredictionResponseSchema,
    status_code=status.HTTP_200_OK,
    response_description="Structured facial expression emotion predictions and bounding boxes",
)
async def create_prediction(
    image: UploadFile = File(..., description="Uploaded image file (JPG, PNG, WEBP)"),
    session_id: uuid.UUID | None = Form(
        default=None, description="Optional associated session UUID"
    ),
    request_id: str = Depends(get_request_id),
    db: AsyncSession = Depends(get_db),
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponseSchema:
    """Ingest image upload, execute Phase 09 ML inference, persist metadata, and return structured prediction."""
    image_bytes = await image.read()
    filename = image.filename or "unknown.jpg"
    content_type = image.content_type

    return await service.predict_and_persist(
        image_bytes=image_bytes,
        filename=filename,
        content_type=content_type,
        db=db,
        request_id=request_id,
        session_id=session_id,
    )


@router.get(
    "/{prediction_id}",
    summary="Retrieve Stored Prediction Details",
    response_model=PredictionDetailSchema,
    status_code=status.HTTP_200_OK,
    response_description="Persisted prediction metadata with detected face results",
)
async def get_prediction(
    prediction_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionDetailSchema:
    """Fetch stored prediction record by UUID."""
    return await service.get_prediction_by_id(db=db, prediction_id=prediction_id)
