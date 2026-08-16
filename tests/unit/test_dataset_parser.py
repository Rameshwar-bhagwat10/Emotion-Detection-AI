"""Unit tests for FER2013 record parser and data models."""

import numpy as np
import pytest

from ml.datasets.fer2013.parser import (
    DatasetRecord,
    parse_csv_row,
    parse_pixels_string,
    pixels_to_hash,
)


def test_parse_pixels_string_valid() -> None:
    """Verify parsing a valid 2304-length string into (48, 48) uint8 array."""
    pixel_vals = [i % 256 for i in range(2304)]
    pixel_str = " ".join(map(str, pixel_vals))

    arr = parse_pixels_string(pixel_str)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (48, 48)
    assert arr.dtype == np.uint8
    assert arr[0, 0] == 0
    assert arr[0, 1] == 1


def test_parse_pixels_string_invalid_length() -> None:
    """Verify error raised when pixel string has incorrect count."""
    with pytest.raises(ValueError, match="Expected 2304 pixels"):
        parse_pixels_string("10 20 30")


def test_parse_pixels_string_non_string() -> None:
    """Verify error raised when input is not a string."""
    with pytest.raises(ValueError, match="Expected string"):
        parse_pixels_string(12345)  # type: ignore[arg-type]


def test_parse_csv_row_valid() -> None:
    """Verify parsing standard CSV row into DatasetRecord."""
    pixel_str = " ".join(["128"] * 2304)
    row = {
        "emotion": "3",
        "pixels": pixel_str,
        "Usage": "Training",
    }

    rec = parse_csv_row(row, record_id=42)
    assert isinstance(rec, DatasetRecord)
    assert rec.record_id == 42
    assert rec.label == 3
    assert rec.label_name == "happy"
    assert rec.split == "train"
    assert rec.shape == (48, 48)
    assert rec.image.mean() == 128.0


def test_parse_csv_row_invalid_label() -> None:
    """Verify error raised for out-of-range label."""
    pixel_str = " ".join(["100"] * 2304)
    row = {
        "emotion": "9",
        "pixels": pixel_str,
        "Usage": "Training",
    }
    with pytest.raises(ValueError, match="out of expected range"):
        parse_csv_row(row, record_id=1)


def test_parse_csv_row_missing_emotion() -> None:
    """Verify error raised when emotion column is missing."""
    row = {
        "pixels": " ".join(["100"] * 2304),
        "Usage": "Training",
    }
    with pytest.raises(ValueError, match="Missing emotion/label"):
        parse_csv_row(row, record_id=1)


def test_dataset_record_md5_hash() -> None:
    """Verify deterministic hash for image array."""
    img1 = np.zeros((48, 48), dtype=np.uint8)
    img2 = np.zeros((48, 48), dtype=np.uint8)
    img3 = np.ones((48, 48), dtype=np.uint8)

    rec1 = DatasetRecord(1, img1, 0, "angry", "train")
    rec2 = DatasetRecord(2, img2, 0, "angry", "val")
    rec3 = DatasetRecord(3, img3, 0, "angry", "train")

    assert rec1.md5_hash == rec2.md5_hash
    assert rec1.md5_hash != rec3.md5_hash
    assert pixels_to_hash(img1) == rec1.md5_hash
