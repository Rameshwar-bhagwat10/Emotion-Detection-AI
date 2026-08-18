"""Unit tests for inference image loading and validation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from ml.inference.image_loader import (
    CorruptedImageError,
    ImageNotFoundError,
    InvalidImageDimensionsError,
    UnsupportedImageFormatError,
    load_image,
)


def test_load_image_from_numpy_array():
    """Verify loading and channel normalization of NumPy arrays."""
    # 3-channel RGB uint8
    arr_3ch = np.random.randint(0, 256, (100, 120, 3), dtype=np.uint8)
    rgb, fmt = load_image(arr_3ch)
    assert rgb.shape == (100, 120, 3)
    assert rgb.dtype == np.uint8
    assert fmt == "NUMPY"

    # 1-channel grayscale uint8
    arr_1ch = np.random.randint(0, 256, (80, 90), dtype=np.uint8)
    rgb_1, fmt_1 = load_image(arr_1ch)
    assert rgb_1.shape == (80, 90, 3)

    # Float array [0.0, 1.0]
    arr_float = np.random.rand(50, 50, 3).astype(np.float32)
    rgb_f, _ = load_image(arr_float)
    assert rgb_f.dtype == np.uint8
    assert rgb_f.max() <= 255


def test_load_image_from_pil(tmp_path: Path):
    """Verify loading from PIL Image."""
    pil_img = Image.new("RGB", (64, 48), color="red")
    rgb, fmt = load_image(pil_img)
    assert rgb.shape == (48, 64, 3)
    assert rgb.dtype == np.uint8


def test_load_image_from_file(tmp_path: Path):
    """Verify loading from valid image files on disk."""
    img_path = tmp_path / "sample.jpg"
    sample = np.full((120, 160, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(img_path), sample)

    rgb, fmt = load_image(img_path)
    assert rgb.shape == (120, 160, 3)
    assert fmt == "JPG"


def test_load_image_from_bytes():
    """Verify decoding from valid image byte buffers."""
    sample = np.full((80, 80, 3), 100, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", sample)
    assert success
    raw_bytes = encoded.tobytes()

    rgb, fmt = load_image(raw_bytes)
    assert rgb.shape == (80, 80, 3)
    assert fmt == "BYTES"


def test_load_image_not_found():
    """Verify error on nonexistent file."""
    with pytest.raises(ImageNotFoundError):
        load_image("nonexistent_image_path_123.jpg")


def test_load_image_unsupported_format(tmp_path: Path):
    """Verify error on unsupported file extension."""
    txt_path = tmp_path / "data.txt"
    txt_path.write_text("not an image")
    with pytest.raises(UnsupportedImageFormatError):
        load_image(txt_path)


def test_load_image_corrupted_bytes():
    """Verify error on corrupted byte buffers."""
    bad_bytes = b"not a real image payload"
    with pytest.raises(CorruptedImageError):
        load_image(bad_bytes)


def test_load_image_oversized():
    """Verify error on dimensions exceeding max_dimension."""
    huge_arr = np.zeros((5000, 100, 3), dtype=np.uint8)
    with pytest.raises(InvalidImageDimensionsError):
        load_image(huge_arr, max_dimension=4096)
