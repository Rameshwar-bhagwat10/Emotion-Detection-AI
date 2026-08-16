"""Unit tests for FER2013 dataset validator."""

import tempfile
from pathlib import Path

import numpy as np

from ml.datasets.fer2013.parser import DatasetRecord
from ml.datasets.fer2013.validator import FER2013Validator


def test_validator_valid_record() -> None:
    """Verify validator accepts a well-formed record."""
    validator = FER2013Validator()
    img = np.zeros((48, 48), dtype=np.uint8)
    rec = DatasetRecord(1, img, 3, "happy", "train")

    is_valid, err = validator.validate_record(rec)
    assert is_valid is True
    assert err is None


def test_validator_invalid_shape_record() -> None:
    """Verify validator rejects record with unexpected shape."""
    validator = FER2013Validator()
    img = np.zeros((50, 50), dtype=np.uint8)
    rec = DatasetRecord(1, img, 3, "happy", "train")

    is_valid, err = validator.validate_record(rec)
    assert is_valid is False
    assert "Invalid image shape" in (err or "")


def test_validator_invalid_label_record() -> None:
    """Verify validator rejects record with out-of-range label."""
    validator = FER2013Validator()
    img = np.zeros((48, 48), dtype=np.uint8)
    rec = DatasetRecord(1, img, 10, "unknown", "train")

    is_valid, err = validator.validate_record(rec)
    assert is_valid is False
    assert "Invalid label" in (err or "")


def test_validator_synthetic_csv_file() -> None:
    """Verify validate_dataset on a synthetic CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "fer2013.csv"
        # Write valid header and 2 valid rows, 1 invalid row
        pixel_str = " ".join(["50"] * 2304)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("emotion,pixels,Usage\n")
            f.write(f"0,{pixel_str},Training\n")
            f.write(f"3,{pixel_str},PublicTest\n")
            f.write("9,invalid_pixels,PrivateTest\n")

        validator = FER2013Validator()
        result = validator.validate_dataset(csv_path)

        assert result.total_records == 3
        assert result.valid_records == 2
        assert result.invalid_records == 1
        assert result.split_counts == {"train": 1, "val": 1}
        assert result.class_counts == {"angry": 1, "happy": 1}


def test_validator_missing_path() -> None:
    """Verify validate_dataset on non-existent path."""
    validator = FER2013Validator()
    result = validator.validate_dataset("path/to/nonexistent/file.csv")
    assert result.is_valid is False
    assert len(result.errors) > 0
