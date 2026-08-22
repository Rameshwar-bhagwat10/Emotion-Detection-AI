"""Pydantic schemas package."""

from app.schemas.common import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
)
from app.schemas.prediction import (
    BoundingBoxSchema,
    FacePredictionSchema,
    ModelInfoSchema,
    PredictionDetailSchema,
    PredictionResponseSchema,
    PredictionTimingSchema,
)
from app.schemas.realtime import (
    ClientConfigMessage,
    ClientFrameMessage,
    DetectedFaceRealtime,
    RealtimeErrorMessage,
    RealtimeMetrics,
    RealtimePredictionResponse,
    RealtimeStatusMessage,
)
from app.schemas.session import (
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
)

__all__ = [
    "BoundingBoxSchema",
    "ClientConfigMessage",
    "ClientFrameMessage",
    "DetectedFaceRealtime",
    "ErrorDetail",
    "ErrorResponse",
    "FacePredictionSchema",
    "HealthResponse",
    "ModelInfoSchema",
    "PredictionDetailSchema",
    "PredictionResponseSchema",
    "PredictionTimingSchema",
    "ReadinessResponse",
    "RealtimeErrorMessage",
    "RealtimeMetrics",
    "RealtimePredictionResponse",
    "RealtimeStatusMessage",
    "SessionCreateRequest",
    "SessionListResponse",
    "SessionResponse",
]
