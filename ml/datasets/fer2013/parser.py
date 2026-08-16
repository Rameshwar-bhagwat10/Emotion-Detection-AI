"""FER2013 Record Parser and Data Models.

Parses raw CSV rows, space-separated pixel strings, or decoded image arrays
into immutable typed DatasetRecord objects.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

EMOTION_LABELS: dict[int, str] = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

EMOTION_NAMES: list[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

SPLIT_MAPPING: dict[str, str] = {
    "training": "train",
    "publictest": "val",
    "privatetest": "test",
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}


@dataclass(frozen=True)
class DatasetRecord:
    """Immutable representation of a single facial expression record."""

    record_id: int | str
    image: np.ndarray  # Shape: (48, 48), dtype: uint8
    label: int  # 0 to 6
    label_name: str  # "angry", "disgust", etc.
    split: str  # "train", "val", "test"

    @property
    def shape(self) -> tuple[int, int]:
        """Return image (height, width)."""
        return self.image.shape  # type: ignore[return-value]

    @property
    def md5_hash(self) -> str:
        """Return deterministic MD5 hash of image pixels for duplicate detection."""
        return hashlib.md5(self.image.tobytes()).hexdigest()

    def __repr__(self) -> str:
        return f"DatasetRecord(id={self.record_id}, label={self.label} ({self.label_name}), split='{self.split}', shape={self.shape})"


def parse_pixels_string(pixels_str: str, expected_size: int = 2304) -> np.ndarray:
    """Parse a space-separated string of 2304 pixel intensity integers into a (48, 48) uint8 array.

    Args:
        pixels_str: Space-delimited pixel values in range 0-255.
        expected_size: Expected total count of pixels (48*48 = 2304).

    Returns:
        np.ndarray: 2D array of shape (48, 48) with uint8 values.

    Raises:
        ValueError: If pixel count differs from expected_size or values are invalid.
    """
    if not isinstance(pixels_str, str):
        raise ValueError(f"Expected string of pixels, got {type(pixels_str).__name__}")

    raw_values = pixels_str.strip().split()
    if len(raw_values) != expected_size:
        raise ValueError(f"Expected {expected_size} pixels, but parsed {len(raw_values)}")

    try:
        pixel_array = np.array([int(v) for v in raw_values], dtype=np.uint8)
    except ValueError as exc:
        raise ValueError(f"Failed to parse pixel values to integers: {exc}") from exc

    return pixel_array.reshape((48, 48))


def parse_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes (e.g. PNG / JPEG) into a (48, 48) grayscale uint8 array.

    Args:
        image_bytes: Raw binary image content.

    Returns:
        np.ndarray: 2D array of shape (48, 48) with uint8 values.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            gray_img = pil_img.convert("L")
            if gray_img.size != (48, 48):
                gray_img = gray_img.resize((48, 48), Image.Resampling.BILINEAR)
            return np.array(gray_img, dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Corrupt or invalid image bytes: {exc}") from exc


def pixels_to_hash(image: np.ndarray) -> str:
    """Compute MD5 hash of raw image pixel bytes for deterministic duplicate detection."""
    return hashlib.md5(image.tobytes()).hexdigest()


def parse_csv_row(  # noqa: C901
    row: Mapping[str, Any] | dict[str, Any] | dict[Any, Any], record_id: int | str = 0
) -> DatasetRecord:
    """Parse a single row from FER2013 CSV or DataFrame into a DatasetRecord.

    Expected CSV columns:
      - 'emotion' (or 'label'): integer 0..6
      - 'pixels' (or 'image'): space-separated string or binary bytes / dict
      - 'Usage' (or 'split'): string e.g. "Training", "PublicTest", "PrivateTest"
    """
    # 1. Parse label
    label_raw = row.get("emotion", row.get("label"))
    if label_raw is None:
        raise ValueError("Missing emotion/label column in row")
    try:
        label = int(label_raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid emotion label value: {label_raw}") from exc

    if label not in EMOTION_LABELS:
        raise ValueError(f"Emotion label {label} is out of expected range [0..6]")

    label_name = EMOTION_LABELS[label]

    # 2. Parse split
    split_raw = str(row.get("Usage", row.get("split", "train"))).strip().lower()
    split = SPLIT_MAPPING.get(split_raw, "train")

    # 3. Parse image
    if "pixels" in row and isinstance(row["pixels"], str):
        image = parse_pixels_string(row["pixels"])
    elif "image" in row:
        img_val = row["image"]
        if isinstance(img_val, dict) and "bytes" in img_val:
            image = parse_image_bytes(img_val["bytes"])
        elif isinstance(img_val, bytes):
            image = parse_image_bytes(img_val)
        elif isinstance(img_val, np.ndarray):
            image = img_val.astype(np.uint8)
            if image.ndim == 3 and image.shape[2] == 3:
                image = image[:, :, 0]
            if image.shape != (48, 48):
                raise ValueError(f"Unexpected image shape: {image.shape}")
        elif isinstance(img_val, str):
            image = parse_pixels_string(img_val)
        else:
            raise ValueError(f"Unsupported image format: {type(img_val).__name__}")
    else:
        raise ValueError("No valid image/pixels field found in record")

    return DatasetRecord(
        record_id=record_id,
        image=image,
        label=label,
        label_name=label_name,
        split=split,
    )
