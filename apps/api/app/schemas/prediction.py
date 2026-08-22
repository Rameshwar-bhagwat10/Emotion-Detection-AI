"""Pydantic schemas for prediction requests, responses, and persistence entities."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class BoundingBoxSchema(BaseModel):
    """Face bounding box coordinates in pixel space."""

    x: int = Field(ge=0, description="Top-left X coordinate")
    y: int = Field(ge=0, description="Top-left Y coordinate")
    width: int = Field(gt=0, description="Bounding box width")
    height: int = Field(gt=0, description="Bounding box height")


class FacePredictionSchema(BaseModel):
    """Per-face emotion detection and classification prediction."""

    face_id: int = Field(ge=1, description="Sequential 1-based face identifier")
    bbox: BoundingBoxSchema = Field(description="Bounding box coordinates")
    detection_confidence: float = Field(
        ge=0.0, le=1.0, description="Face detector confidence score [0.0, 1.0]"
    )
    emotion: str = Field(description="Predicted emotion class or 'uncertain'")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Predicted emotion softmax confidence score [0.0, 1.0]"
    )
    is_uncertain: bool = Field(
        default=False, description="True if top confidence was below application threshold"
    )
    probabilities: dict[str, float] = Field(
        description="Softmax probability distribution across all 7 emotions"
    )


class PredictionTimingSchema(BaseModel):
    """Fine-grained inference stage latency breakdown in milliseconds."""

    image_loading_ms: float = Field(ge=0.0, default=0.0)
    face_detection_ms: float = Field(ge=0.0, default=0.0)
    preprocessing_ms: float = Field(ge=0.0, default=0.0)
    inference_ms: float = Field(ge=0.0, default=0.0)
    postprocessing_ms: float = Field(ge=0.0, default=0.0)
    total_ms: float = Field(ge=0.0, default=0.0)


class ModelInfoSchema(BaseModel):
    """Metadata identifying the active ML model."""

    model_name: str = Field(description="Champion model identifier")
    architecture: str = Field(description="Underlying neural network architecture")
    optimization_type: str = Field(default="pruning_30pct", description="Optimization technique")
    device: str = Field(description="Execution device (cpu or cuda)")


class PredictionResponseSchema(BaseModel):
    """Response returned upon successful execution of POST /api/v1/predictions."""

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(
        description="Inference status: success, no_face_detected, invalid_image, error"
    )
    request_id: str = Field(description="Unique correlation request ID")
    prediction_id: str | None = Field(default=None, description="Persisted prediction record UUID")
    session_id: str | None = Field(default=None, description="Optional associated session UUID")
    faces_detected: int = Field(ge=0, description="Number of valid faces detected in the image")
    faces: list[FacePredictionSchema] = Field(
        default_factory=list, description="List of detected face predictions"
    )
    timing: PredictionTimingSchema = Field(description="Pipeline timing metrics")
    model_info: ModelInfoSchema = Field(description="Active model metadata")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Prediction timestamp"
    )


class PredictionDetailSchema(BaseModel):
    """Detailed stored prediction retrieved via GET /api/v1/predictions/{id}."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(description="Prediction record UUID")
    session_id: uuid.UUID | None = Field(default=None, description="Associated session UUID")
    request_id: str = Field(description="Original request ID")
    model_version: str = Field(description="Model version used during inference")
    status: str = Field(description="Inference outcome status")
    faces_detected: int = Field(description="Count of detected faces")
    processing_time_ms: float = Field(description="Total processing time in milliseconds")
    image_width: int | None = Field(default=None)
    image_height: int | None = Field(default=None)
    created_at: datetime = Field(description="Timestamp of prediction")
    faces: list[FacePredictionSchema] = Field(default_factory=list)
