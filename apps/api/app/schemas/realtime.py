"""Pydantic schemas for real-time WebSocket communication, frame exchange, and temporal predictions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.prediction import BoundingBoxSchema


class DetectedFaceRealtime(BaseModel):
    """Real-time detected face representation with temporal smoothing and raw metrics."""

    model_config = ConfigDict(extra="ignore")

    face_id: int = Field(
        ...,
        description="Session-local tracking index for the face (e.g. 1, 2)",
        ge=1,
    )
    bbox: BoundingBoxSchema = Field(
        ..., description="Pixel bounding box in the analyzed video frame"
    )
    detection_confidence: float = Field(
        ..., description="Face detector confidence score", ge=0.0, le=1.0
    )
    emotion: str = Field(
        ..., description="Active/smoothed dominant emotion label (e.g., happy, neutral)"
    )
    confidence: float = Field(
        ..., description="Active/smoothed dominant emotion confidence", ge=0.0, le=1.0
    )
    is_uncertain: bool = Field(
        default=False,
        description="Flag indicating if the prediction confidence is below threshold",
    )
    probabilities: dict[str, float] = Field(
        ..., description="Smoothed 7-class emotion probability distribution"
    )
    raw_emotion: str = Field(
        ..., description="Unsmoothed instantaneous emotion label from Phase 09 inference"
    )
    raw_confidence: float = Field(
        ...,
        description="Unsmoothed instantaneous confidence from Phase 09 inference",
        ge=0.0,
        le=1.0,
    )


class RealtimeMetrics(BaseModel):
    """Performance telemetry for a single real-time inference cycle."""

    model_config = ConfigDict(extra="ignore")

    inference_time_ms: float = Field(
        ..., description="Phase 09 neural network & face detection latency in milliseconds"
    )
    total_processing_time_ms: float = Field(
        ..., description="Total pipeline latency (decode + infer + smooth) in milliseconds"
    )
    fps: float = Field(
        default=0.0, description="Measured server inference processing frames-per-second"
    )
    dropped_frames: int = Field(
        default=0, description="Total number of dropped frames due to backpressure in this stream"
    )
    server_timestamp: float = Field(
        ..., description="Unix epoch timestamp when response was generated on server"
    )


class RealtimePredictionResponse(BaseModel):
    """Server-to-client WebSocket payload delivering frame emotion predictions."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["prediction"] = "prediction"
    frame_id: int = Field(
        ..., description="Correlated frame ID matching the client's frame request"
    )
    client_timestamp: float | None = Field(
        default=None, description="Echoed timestamp provided by the client upon capture"
    )
    session_id: str | None = Field(
        default=None, description="UUID of the active real-time analysis session"
    )
    faces_detected: int = Field(
        ..., description="Total number of valid faces detected in this frame", ge=0
    )
    faces: list[DetectedFaceRealtime] = Field(
        default_factory=list, description="Array of detected faces and their emotion analytics"
    )
    metrics: RealtimeMetrics = Field(
        ..., description="Inference performance telemetry and frame rate statistics"
    )


class RealtimeStatusMessage(BaseModel):
    """Server-to-client message communicating connection and session lifecycle state."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["status"] = "status"
    status: str = Field(
        ..., description="Status string: connected, active, paused, stopping, closed"
    )
    session_id: str | None = Field(default=None, description="UUID of associated analysis session")
    message: str | None = Field(
        default=None, description="Human-readable informational status message"
    )
    details: dict[str, Any] | None = Field(
        default=None, description="Additional environment or configuration details"
    )


class RealtimeErrorMessage(BaseModel):
    """Server-to-client structured error message for protocol or inference failures."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["error"] = "error"
    code: str = Field(
        ...,
        description="Standardized error code (e.g., INVALID_FRAME, SERVER_BUSY, INFERENCE_ERROR)",
    )
    message: str = Field(..., description="User-safe error description")
    frame_id: int | None = Field(
        default=None, description="Optional correlated frame ID where error occurred"
    )


class ClientFrameMessage(BaseModel):
    """Client-to-server JSON frame message payload (used when frames are transmitted via JSON/base64)."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["frame"] = "frame"
    frame_id: int = Field(..., description="Monotonically increasing frame sequence identifier")
    timestamp: float = Field(..., description="Client capture timestamp in milliseconds or seconds")
    image: str = Field(..., description="Base64-encoded image data string")


class ClientConfigMessage(BaseModel):
    """Client-to-server configuration update for stream tuning."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["config"] = "config"
    target_fps: int | None = Field(
        default=None, description="Requested target processing frame rate", ge=1, le=60
    )
    smoothing_enabled: bool | None = Field(
        default=None, description="Enable or disable exponential temporal smoothing"
    )
    confidence_threshold: float | None = Field(
        default=None,
        description="Custom confidence threshold for low-confidence marking",
        ge=0.0,
        le=1.0,
    )
