"""Structured data schemas and output models for the emotion inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class InferenceStatus(StrEnum):
    """Controlled status codes for inference pipeline responses."""

    SUCCESS = "SUCCESS"
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    INVALID_IMAGE = "INVALID_IMAGE"
    INVALID_INPUT = "INVALID_INPUT"
    INFERENCE_ERROR = "INFERENCE_ERROR"


@dataclass
class BoundingBoxDict:
    """Bounding box coordinates and dimensions."""

    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
        }


@dataclass
class FacePrediction:
    """Inference emotion prediction for a single detected face."""

    face_id: int
    bbox: BoundingBoxDict
    detection_confidence: float
    emotion: str
    confidence: float
    is_uncertain: bool
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "face_id": self.face_id,
            "bbox": self.bbox.to_dict(),
            "detection_confidence": round(float(self.detection_confidence), 4),
            "emotion": self.emotion,
            "confidence": round(float(self.confidence), 4),
            "is_uncertain": self.is_uncertain,
            "probabilities": {k: round(float(v), 4) for k, v in self.probabilities.items()},
        }


@dataclass
class InferenceTiming:
    """Latency breakdown for inference pipeline stages in milliseconds."""

    image_loading_ms: float = 0.0
    face_detection_ms: float = 0.0
    preprocessing_ms: float = 0.0
    inference_ms: float = 0.0
    postprocessing_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "image_loading_ms": round(self.image_loading_ms, 2),
            "face_detection_ms": round(self.face_detection_ms, 2),
            "preprocessing_ms": round(self.preprocessing_ms, 2),
            "inference_ms": round(self.inference_ms, 2),
            "postprocessing_ms": round(self.postprocessing_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


@dataclass
class ImageInfo:
    """Metadata regarding the input image."""

    width: int
    height: int
    channels: int
    format: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "format": self.format,
        }


@dataclass
class ModelInfo:
    """Information regarding the underlying emotion recognition model."""

    model_name: str
    architecture: str
    optimization_type: str
    device: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "optimization_type": self.optimization_type,
            "device": self.device,
        }


@dataclass
class ImageInferenceResult:
    """Complete structured prediction result for a single image."""

    status: InferenceStatus
    image: ImageInfo | None = None
    faces_detected: int = 0
    faces: list[FacePrediction] = field(default_factory=list)
    timing: InferenceTiming | None = None
    model_info: ModelInfo | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert entire result to a clean serializable dictionary."""
        return {
            "status": self.status.value,
            "image": self.image.to_dict() if self.image else None,
            "faces_detected": self.faces_detected,
            "faces": [f.to_dict() for f in self.faces],
            "timing": self.timing.to_dict() if self.timing else None,
            "model_info": self.model_info.to_dict() if self.model_info else None,
            "error_message": self.error_message,
        }
