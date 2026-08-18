"""Emotion Detection AI — Production Inference & Prediction Pipeline package."""

from __future__ import annotations

from ml.inference.confidence import (
    apply_confidence_gate,
    compute_confidence,
    validate_probabilities,
)
from ml.inference.config import (
    DeviceConfig,
    FaceDetectionConfig,
    InferencePipelineConfig,
    ModelInferenceConfig,
    load_inference_config,
)
from ml.inference.engine import EmotionInferenceEngine
from ml.inference.face_detector import (
    BaseFaceDetector,
    FaceBoundingBox,
    FaceDetection,
    HaarCascadeFaceDetector,
    PassThroughFaceDetector,
    YuNetFaceDetector,
    create_face_detector,
)
from ml.inference.image_loader import (
    CorruptedImageError,
    ImageLoadError,
    ImageNotFoundError,
    InvalidImageDimensionsError,
    UnsupportedImageFormatError,
    load_image,
)
from ml.inference.model_loader import (
    ModelLoadingError,
    ModelManager,
    load_champion_model,
    resolve_device,
)
from ml.inference.postprocessing import process_logits
from ml.inference.predictor import EmotionPredictor
from ml.inference.preprocessor import FacePreprocessor, InvalidCropError, PreprocessingError
from ml.inference.schemas import (
    BoundingBoxDict,
    FacePrediction,
    ImageInferenceResult,
    ImageInfo,
    InferenceStatus,
    InferenceTiming,
    ModelInfo,
)

__all__ = [
    "DeviceConfig",
    "ModelInferenceConfig",
    "FaceDetectionConfig",
    "InferencePipelineConfig",
    "load_inference_config",
    "compute_confidence",
    "apply_confidence_gate",
    "validate_probabilities",
    "BaseFaceDetector",
    "FaceBoundingBox",
    "FaceDetection",
    "HaarCascadeFaceDetector",
    "YuNetFaceDetector",
    "PassThroughFaceDetector",
    "create_face_detector",
    "ImageLoadError",
    "ImageNotFoundError",
    "UnsupportedImageFormatError",
    "CorruptedImageError",
    "InvalidImageDimensionsError",
    "load_image",
    "ModelLoadingError",
    "ModelManager",
    "load_champion_model",
    "resolve_device",
    "FacePreprocessor",
    "PreprocessingError",
    "InvalidCropError",
    "process_logits",
    "EmotionInferenceEngine",
    "EmotionPredictor",
    "InferenceStatus",
    "BoundingBoxDict",
    "FacePrediction",
    "InferenceTiming",
    "ImageInfo",
    "ModelInfo",
    "ImageInferenceResult",
]
