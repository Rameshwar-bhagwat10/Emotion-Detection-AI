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

    def crop_face(
        self,
        image_rgb: np.ndarray,
        bbox: FaceBoundingBox,
        square_crop: bool = True,
    ) -> np.ndarray:
        """Extract a 2D face region from an RGB image.

        Args:
            image_rgb: Full RGB image of shape [H, W, 3].
            bbox: Bounding box for the face.
            square_crop: Whether to square-adjust bounding box to preserve facial aspect ratio.

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

        if square_crop:
            # Center and square-adjust bounding box to preserve natural facial aspect ratio
            side = max(bbox.width, bbox.height)
            cx = bbox.x + bbox.width // 2
            cy = bbox.y + bbox.height // 2

            x1 = cx - side // 2
            y1 = cy - side // 2
            x2 = x1 + side
            y2 = y1 + side

            # Shift window if extending past frame boundaries
            if x1 < 0:
                x2 = min(w, x2 - x1)
                x1 = 0
            if y1 < 0:
                y2 = min(h, y2 - y1)
                y1 = 0
            if x2 > w:
                x1 = max(0, x1 - (x2 - w))
                x2 = w
            if y2 > h:
                y1 = max(0, y1 - (y2 - h))
                y2 = h
        else:
            x1 = bbox.x
            y1 = bbox.y
            x2 = bbox.x + bbox.width
            y2 = bbox.y + bbox.height

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        crop = image_rgb[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            raise InvalidCropError(f"Extracted face crop is empty for bbox {bbox.to_tuple()}")

        return crop

    def preprocess_single_crop(self, crop_rgb: np.ndarray) -> torch.Tensor:
        """Transform a single RGB face crop into a normalized float32 tensor of shape [C, H, W].

        Pipeline (3-channel RGB):
            1. Ensure RGB 3-channel format
            2. Resize to (target_w, target_h) (e.g. 112x112)
            3. Scale uint8 [0, 255] -> float32 [0.0, 1.0]
            4. Normalize with ImageNet or configured mean/std
            5. Convert to PyTorch Tensor [3, H, W]

        Pipeline (1-channel Grayscale):
            1. Convert RGB to Grayscale
            2. Resize to (target_w, target_h) (e.g. 48x48)
            3. Scale uint8 [0, 255] -> float32 [0.0, 1.0]
            4. Normalize with training mean/std
            5. Convert to PyTorch Tensor [1, H, W]

        Args:
            crop_rgb: Face crop array [crop_h, crop_w, channels] uint8.

        Returns:
            Normalized tensor of shape [C, target_h, target_w] (float32).
        """
        if crop_rgb.size == 0:
            raise InvalidCropError("Cannot preprocess empty face crop.")

        interp = cv2.INTER_AREA if (crop_rgb.shape[0] > self.target_h or crop_rgb.shape[1] > self.target_w) else cv2.INTER_LINEAR

        if self.config.input_channels == 3:
            # 1. Ensure 3-channel RGB
            if crop_rgb.ndim == 3 and crop_rgb.shape[2] == 3:
                rgb = crop_rgb
            elif crop_rgb.ndim == 2:
                rgb = cv2.cvtColor(crop_rgb, cv2.COLOR_GRAY2RGB)
            elif crop_rgb.ndim == 3 and crop_rgb.shape[2] == 1:
                rgb = cv2.cvtColor(crop_rgb.squeeze(axis=-1), cv2.COLOR_GRAY2RGB)
            elif crop_rgb.ndim == 3 and crop_rgb.shape[2] == 4:
                rgb = cv2.cvtColor(crop_rgb, cv2.COLOR_RGBA2RGB)
            else:
                raise PreprocessingError(f"Unexpected crop array shape: {crop_rgb.shape}")

            # 2. Resize to model input dimensions
            resized = cv2.resize(rgb, (self.target_w, self.target_h), interpolation=interp)

            # 3. Scale [0, 255] -> [0.0, 1.0]
            scaled = resized.astype(np.float32) / 255.0

            # 4. Normalize with mean and std (broadcast over H, W, 3)
            mean = self.mean if len(self.mean) == 3 else np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = self.std if len(self.std) == 3 else np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (scaled - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)

            # 5. Permute to [3, H, W]
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).to(dtype=torch.float32)
            return tensor
        else:
            # 1. Convert to Grayscale if 3-channel
            if crop_rgb.ndim == 3 and crop_rgb.shape[2] == 3:
                gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
            elif crop_rgb.ndim == 2:
                gray = crop_rgb
            elif crop_rgb.ndim == 3 and crop_rgb.shape[2] == 1:
                gray = crop_rgb.squeeze(axis=-1)
            elif crop_rgb.ndim == 3 and crop_rgb.shape[2] == 4:
                gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGBA2GRAY)
            else:
                raise PreprocessingError(f"Unexpected crop array shape: {crop_rgb.shape}")

            # 2. Resize to model input dimensions
            resized = cv2.resize(gray, (self.target_w, self.target_h), interpolation=interp)

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
