"""Convenience predictor interfaces and pipeline wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image

from ml.inference.config import InferencePipelineConfig
from ml.inference.engine import EmotionInferenceEngine
from ml.inference.schemas import FacePrediction, ImageInferenceResult


class EmotionPredictor:
    """Convenience wrapper around EmotionInferenceEngine for simplified caller usage."""

    def __init__(
        self,
        config: InferencePipelineConfig | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """Initialize EmotionPredictor."""
        self.engine = EmotionInferenceEngine(config=config, config_path=config_path, auto_load=True)

    def predict(
        self,
        image_source: str | Path | bytes | BinaryIO | Image.Image | np.ndarray,
    ) -> ImageInferenceResult:
        """Predict emotions for all faces in an image."""
        return self.engine.predict_image(image_source)

    def predict_batch(
        self,
        image_sources: Sequence[str | Path | bytes | BinaryIO | Image.Image | np.ndarray],
    ) -> list[ImageInferenceResult]:
        """Predict emotions across multiple images."""
        return self.engine.predict_batch(image_sources)

    def predict_single_face_crop(self, face_crop_rgb: np.ndarray) -> FacePrediction:
        """Directly predict emotion on an already-cropped RGB face array."""
        result = self.engine.predict_image(face_crop_rgb)
        if not result.faces:
            raise ValueError(f"No face detected in crop (status: {result.status})")
        return result.faces[0]


__all__ = ["EmotionPredictor", "EmotionInferenceEngine"]
