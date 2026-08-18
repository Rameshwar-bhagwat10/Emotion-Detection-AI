"""Deterministic face crop extraction and preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch

from ml.inference.config import ModelInferenceConfig
from ml.inference.face_detector import FaceBoundingBox


class PreprocessingError(Exception):
    """Base exception for face cropping and preprocessing errors."""

    pass


class InvalidCropError(PreprocessingError):
    """Raised when extracted face crop has zero area or invalid pixel data."""

    pass


class FacePreprocessor:
    """Preprocesses cropped faces into normalized PyTorch tensors matching model training."""

    def __init__(self, config: ModelInferenceConfig | None = None) -> None:
        """Initialize face preprocessor with model configuration.

        Args:
            config: Optional ModelInferenceConfig instance.
        """
        self.config = config or ModelInferenceConfig()
        self.target_h, self.target_w = self.config.input_size
        self.mean = np.array(self.config.mean, dtype=np.float32)
        self.std = np.array(self.config.std, dtype=np.float32)

    def crop_face(self, image_rgb: np.ndarray, bbox: FaceBoundingBox) -> np.ndarray:
        """Extract a 2D face region from an RGB image.

        Args:
            image_rgb: Full RGB image of shape [H, W, 3].
            bbox: Bounding box for the face.

        Returns:
            Extracted RGB face crop array of shape [crop_h, crop_w, 3] uint8.
        """
        if bbox.width <= 0 or bbox.height <= 0 or bbox.area <= 0:
            raise InvalidCropError(
                f"Bounding box has zero or negative dimension: {bbox.to_tuple()}"
            )

        h, w = image_rgb.shape[:2]
        if bbox.x >= w or bbox.y >= h or bbox.x + bbox.width <= 0 or bbox.y + bbox.height <= 0:
            raise InvalidCropError(
                f"Bounding box {bbox.to_tuple()} is entirely outside image bounds ({w}x{h})"
            )

        x1 = max(0, min(bbox.x, w - 1))
        y1 = max(0, min(bbox.y, h - 1))
        x2 = max(x1 + 1, min(bbox.x + bbox.width, w))
        y2 = max(y1 + 1, min(bbox.y + bbox.height, h))

        crop = image_rgb[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            raise InvalidCropError(f"Extracted face crop is empty for bbox {bbox.to_tuple()}")

        return crop

    def preprocess_single_crop(self, crop_rgb: np.ndarray) -> torch.Tensor:
        """Transform a single RGB face crop into a normalized float32 tensor of shape [1, H, W].

        Pipeline:
            1. Convert RGB to Grayscale
            2. Resize to (target_w, target_h) (e.g. 48x48)
            3. Scale uint8 [0, 255] -> float32 [0.0, 1.0]
            4. Normalize with training mean and standard deviation: (x - mean) / std
            5. Convert to PyTorch Tensor [1, H, W]

        Args:
            crop_rgb: Face crop array [crop_h, crop_w, 3] uint8.

        Returns:
            Normalized tensor of shape [1, target_h, target_w] (float32).
        """
        if crop_rgb.size == 0:
            raise InvalidCropError("Cannot preprocess empty face crop.")

        # 1. Convert to Grayscale if 3-channel
        if crop_rgb.ndim == 3 and crop_rgb.shape[2] == 3:
            gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        elif crop_rgb.ndim == 2:
            gray = crop_rgb
        elif crop_rgb.ndim == 3 and crop_rgb.shape[2] == 1:
            gray = crop_rgb.squeeze(axis=-1)
        else:
            raise PreprocessingError(f"Unexpected crop array shape: {crop_rgb.shape}")

        # 2. Resize to model input dimensions
        resized = cv2.resize(gray, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

        # 3. Scale [0, 255] -> [0.0, 1.0]
        scaled = resized.astype(np.float32) / 255.0

        # 4. Normalize with training split stats
        normalized = (scaled - self.mean[0]) / self.std[0]

        # 5. Convert to PyTorch tensor [1, H, W]
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(dtype=torch.float32)
        return tensor

    def preprocess_batch(
        self,
        crops: Sequence[np.ndarray],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Preprocess a sequence of face crops into a batched tensor [B, 1, H, W].

        Args:
            crops: Sequence of RGB face crops.
            device: Target torch device.

        Returns:
            Batched PyTorch tensor of shape [B, 1, H, W].
        """
        if not crops:
            raise PreprocessingError("No face crops provided for batch preprocessing.")

        tensors = [self.preprocess_single_crop(crop) for crop in crops]
        batch_tensor = torch.stack(tensors, dim=0).to(device=device)
        return batch_tensor
