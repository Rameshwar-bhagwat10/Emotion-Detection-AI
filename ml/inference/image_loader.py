"""Image ingestion, validation, and standardization service for the inference pipeline."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ImageLoadError(Exception):
    """Base exception for image loading failures."""

    pass


class ImageNotFoundError(ImageLoadError):
    """Raised when an image file cannot be found."""

    pass


class UnsupportedImageFormatError(ImageLoadError):
    """Raised when an unsupported image extension/format is supplied."""

    pass


class CorruptedImageError(ImageLoadError):
    """Raised when image bytes/file cannot be decoded."""

    pass


class InvalidImageDimensionsError(ImageLoadError):
    """Raised when image dimensions are out of allowable bounds."""

    pass


def _load_from_numpy(img_arr: np.ndarray) -> np.ndarray:
    """Normalize numpy array input to RGB [H, W, 3] uint8."""
    if img_arr.size == 0:
        raise CorruptedImageError("Empty numpy array supplied as image source.")

    if img_arr.ndim == 2:
        rgb_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
    elif img_arr.ndim == 3 and img_arr.shape[2] == 1:
        rgb_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
    elif img_arr.ndim == 3 and img_arr.shape[2] == 3:
        rgb_arr = img_arr.copy()
    elif img_arr.ndim == 3 and img_arr.shape[2] == 4:
        rgb_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
    else:
        raise CorruptedImageError(f"Unsupported image array shape: {img_arr.shape}")

    if rgb_arr.dtype != np.uint8:
        if rgb_arr.max() <= 1.0 and rgb_arr.min() >= 0.0:
            rgb_arr = (rgb_arr * 255.0).astype(np.uint8)
        else:
            rgb_arr = np.clip(rgb_arr, 0, 255).astype(np.uint8)

    return rgb_arr


def _load_from_path(p: Path) -> tuple[np.ndarray, str]:
    """Read and decode image file from filesystem path."""
    if not p.exists() or not p.is_file():
        raise ImageNotFoundError(f"Image file not found: {p}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedImageFormatError(
            f"Unsupported image extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    fmt = ext.lstrip(".").upper()

    try:
        with open(p, "rb") as f:
            raw_bytes = f.read()
    except Exception as e:
        raise CorruptedImageError(f"Failed to read image file: {e}") from e

    if not raw_bytes:
        raise CorruptedImageError(f"Image file is empty: {p}")

    np_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise CorruptedImageError(f"Failed to decode image file: {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), fmt


def _load_from_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode image from raw byte buffer."""
    if not raw_bytes:
        raise CorruptedImageError("Empty bytes buffer provided.")

    np_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise CorruptedImageError("Failed to decode image from raw bytes.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_image(
    image_source: str | Path | bytes | BinaryIO | Image.Image | np.ndarray,
    max_dimension: int = 4096,
) -> tuple[np.ndarray, str]:
    """Load, validate, and standardize any image input into a uint8 RGB numpy array.

    Args:
        image_source: Path to image file, raw bytes, file-like stream, PIL Image, or numpy array.
        max_dimension: Maximum allowed width or height in pixels.

    Returns:
        Tuple of (standardized_rgb_array [H, W, 3] uint8, format_string).
    """
    if isinstance(image_source, np.ndarray):
        rgb_arr = _load_from_numpy(image_source)
        fmt = "NUMPY"
    elif isinstance(image_source, Image.Image):
        fmt = image_source.format or "PIL"
        rgb_arr = np.array(image_source.convert("RGB"), dtype=np.uint8)
    elif isinstance(image_source, (str, Path)):
        rgb_arr, fmt = _load_from_path(Path(image_source))
    elif isinstance(image_source, (bytes, bytearray)):
        rgb_arr = _load_from_bytes(bytes(image_source))
        fmt = "BYTES"
    elif isinstance(image_source, (io.BytesIO, BinaryIO)):
        rgb_arr = _load_from_bytes(image_source.read())
        fmt = "BYTES"
    else:
        raise ImageLoadError(f"Unsupported image source type: {type(image_source)}")

    h, w = rgb_arr.shape[:2]
    if h <= 0 or w <= 0:
        raise InvalidImageDimensionsError(f"Invalid image dimensions: {w}x{h}")
    if h > max_dimension or w > max_dimension:
        raise InvalidImageDimensionsError(
            f"Image dimensions {w}x{h} exceed maximum allowable dimension {max_dimension}"
        )

    return rgb_arr, fmt
