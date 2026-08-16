"""FER2013 Dataset Validator.

Performs thorough structural and record-level integrity validation of the dataset.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ml.datasets.fer2013.parser import (
    EMOTION_LABELS,
    DatasetRecord,
    parse_csv_row,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetValidationResult:
    """Structured validation outcome of the FER2013 dataset."""

    dataset_path: str
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    missing_records: int = 0
    invalid_labels: list[int | str] = field(default_factory=list)
    invalid_dimensions: int = 0
    corrupt_images: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    is_valid: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert validation result to dictionary format."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize validation result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class FER2013Validator:
    """Validator for FER2013 raw data files and record objects."""

    def __init__(self, expected_shape: tuple[int, int] = (48, 48)) -> None:
        """Initialize validator with expected dimensions.

        Args:
            expected_shape: Tuple of expected (height, width) = (48, 48).
        """
        self.expected_shape = expected_shape

    def validate_record(self, record: DatasetRecord) -> tuple[bool, str | None]:
        """Validate an individual parsed record.

        Returns:
            tuple[bool, str | None]: (is_valid, error_reason)
        """
        # 1. Validate label
        if record.label not in EMOTION_LABELS:
            return False, f"Invalid label {record.label}: must be in 0..6"

        # 2. Validate label name consistency
        if EMOTION_LABELS[record.label] != record.label_name:
            return (
                False,
                f"Label mismatch: {record.label} != '{record.label_name}'",
            )

        # 3. Validate image dimensions
        if record.shape != self.expected_shape:
            return (
                False,
                f"Invalid image shape {record.shape}: expected {self.expected_shape}",
            )

        # 4. Validate pixel range
        if record.image.min() < 0 or record.image.max() > 255 or record.image.dtype != "uint8":
            return False, "Pixel values out of valid uint8 [0, 255] range"

        # 5. Validate split
        if record.split not in {"train", "val", "test"}:
            return False, f"Invalid split '{record.split}'"

        return True, None

    def validate_dataset(self, data_path: str | Path) -> DatasetValidationResult:  # noqa: C901
        """Perform comprehensive validation of the dataset file.

        Args:
            data_path: Path to dataset file or folder.

        Returns:
            DatasetValidationResult: Detailed validation outcome.
        """
        path = Path(data_path)
        result = DatasetValidationResult(dataset_path=str(path))

        # Check path existence
        if not path.exists():
            result.errors.append(f"Path does not exist: {path}")
            result.is_valid = False
            return result

        target_file = path / "fer2013.csv" if path.is_dir() else path
        if not target_file.exists() or not target_file.is_file():
            result.errors.append(f"Target dataset file not found: {target_file}")
            result.is_valid = False
            return result

        # Read and validate line-by-line
        try:
            with open(target_file, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []

                # Validate expected headers
                has_emotion = any(h.lower() in ["emotion", "label"] for h in headers)
                has_pixels = any(h.lower() in ["pixels", "image"] for h in headers)
                if not (has_emotion and has_pixels):
                    result.errors.append(f"Missing required CSV headers. Found: {headers}")
                    result.is_valid = False
                    return result

                for idx, row in enumerate(reader):
                    result.total_records += 1
                    try:
                        record = parse_csv_row(row, record_id=idx)
                        is_rec_valid, err = self.validate_record(record)
                        if is_rec_valid:
                            result.valid_records += 1
                            # Update split count
                            result.split_counts[record.split] = (
                                result.split_counts.get(record.split, 0) + 1
                            )
                            # Update class count
                            result.class_counts[record.label_name] = (
                                result.class_counts.get(record.label_name, 0) + 1
                            )
                        else:
                            result.invalid_records += 1
                            if "shape" in (err or ""):
                                result.invalid_dimensions += 1
                            elif "label" in (err or ""):
                                result.invalid_labels.append(record.label)
                            result.errors.append(f"Row {idx}: {err}")
                    except Exception as exc:
                        result.invalid_records += 1
                        result.corrupt_images += 1
                        result.errors.append(f"Row {idx}: Failed to parse record: {exc}")

        except Exception as exc:
            result.errors.append(f"Failed to read dataset file: {exc}")
            result.is_valid = False
            return result

        # Verify final status
        if result.total_records == 0:
            result.errors.append("Dataset file is empty.")
            result.is_valid = False
        elif result.invalid_records > 0:
            result.warnings.append(
                f"Found {result.invalid_records} invalid records out of {result.total_records}."
            )
            result.is_valid = result.valid_records > 0
        else:
            result.is_valid = True

        logger.info(
            "Validation finished: total=%d, valid=%d, invalid=%d",
            result.total_records,
            result.valid_records,
            result.invalid_records,
        )
        return result
