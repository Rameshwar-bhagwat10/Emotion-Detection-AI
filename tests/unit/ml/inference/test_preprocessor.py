"""Unit tests for face crop extraction and deterministic preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.inference.config import ModelInferenceConfig
from ml.inference.face_detector import FaceBoundingBox
from ml.inference.preprocessor import FacePreprocessor, InvalidCropError, PreprocessingError


def test_crop_face_valid():
    """Verify cropping rectangular region from RGB image."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[40:100, 50:120] = 200  # Draw white rectangle

    preprocessor = FacePreprocessor()
    bbox = FaceBoundingBox(x=50, y=40, width=70, height=60)
    crop = preprocessor.crop_face(img, bbox)

    assert crop.shape == (60, 70, 3)
    assert crop.dtype == np.uint8
    assert (crop == 200).all()


def test_crop_face_invalid_raises():
    """Verify raising InvalidCropError on invalid/empty coordinates."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    preprocessor = FacePreprocessor()
    bad_box = FaceBoundingBox(x=100, y=100, width=0, height=0)

    with pytest.raises(InvalidCropError):
        preprocessor.crop_face(img, bad_box)


def test_preprocess_single_crop():
    """Verify conversion to normalized [1, 48, 48] float32 tensor."""
    crop = np.full((80, 80, 3), 128, dtype=np.uint8)
    cfg = ModelInferenceConfig(input_size=[48, 48], mean=[0.507743], std=[0.255009])
    preprocessor = FacePreprocessor(config=cfg)

    tensor = preprocessor.preprocess_single_crop(crop)
    assert tensor.shape == (1, 48, 48)
    assert tensor.dtype == torch.float32

    # Verify normalization calculation: (128/255.0 - 0.507743) / 0.255009
    expected_val = (128.0 / 255.0 - 0.507743) / 0.255009
    assert abs(tensor[0, 0, 0].item() - expected_val) < 1e-4


def test_preprocess_batch():
    """Verify batched tensor construction of shape [B, 1, 48, 48]."""
    crops = [
        np.full((60, 60, 3), 100, dtype=np.uint8),
        np.full((70, 70, 3), 200, dtype=np.uint8),
        np.full((50, 50, 3), 50, dtype=np.uint8),
    ]
    preprocessor = FacePreprocessor()
    batch = preprocessor.preprocess_batch(crops, device="cpu")

    assert batch.shape == (3, 1, 48, 48)
    assert batch.dtype == torch.float32
    assert batch.device.type == "cpu"


def test_preprocess_batch_empty_raises():
    """Verify error when empty sequence of crops is passed."""
    preprocessor = FacePreprocessor()
    with pytest.raises(PreprocessingError):
        preprocessor.preprocess_batch([])
