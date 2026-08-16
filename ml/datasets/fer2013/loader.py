"""FER2013 Dataset Loader.

Provides efficient, memory-safe loading of FER2013 records from CSV or Parquet files.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from ml.datasets.fer2013.parser import DatasetRecord, parse_csv_row

logger = logging.getLogger(__name__)


class FER2013Loader:
    """Dataset loader for FER2013 facial expression records."""

    def __init__(self, data_path: str | Path = "data/raw/fer2013/fer2013.csv") -> None:
        """Initialize loader with path to dataset file or raw directory.

        Args:
            data_path: Path to `fer2013.csv` or directory containing raw FER2013 files.
        """
        self.data_path = Path(data_path)
        self._resolved_file: Path | None = None

    def resolve_dataset_file(self) -> Path:
        """Locate and return the primary dataset file."""
        if self._resolved_file and self._resolved_file.exists():
            return self._resolved_file

        if self.data_path.is_file() and self.data_path.exists():
            self._resolved_file = self.data_path
            return self._resolved_file

        if self.data_path.is_dir():
            # Check for standard fer2013.csv
            csv_candidate = self.data_path / "fer2013.csv"
            if csv_candidate.is_file():
                self._resolved_file = csv_candidate
                return self._resolved_file

            # Check for parquet files
            parquet_files = list(self.data_path.glob("*.parquet"))
            if parquet_files:
                self._resolved_file = parquet_files[0]
                return self._resolved_file

        raise FileNotFoundError(
            f"FER2013 dataset not found at '{self.data_path}'. "
            "Please run 'python scripts/data/download-dataset.py' to acquire the dataset."
        )

    def count_records(self) -> int:
        """Return total record count without loading all images into memory."""
        file_path = self.resolve_dataset_file()
        if file_path.suffix.lower() == ".csv":
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                # Subtract header line
                return max(0, sum(1 for _ in f) - 1)
        elif file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path, columns=["label"])
            return len(df)
        return 0

    def stream_records(self, split: str | None = None) -> Generator[DatasetRecord, None, None]:
        """Stream dataset records lazily to minimize memory footprint.

        Args:
            split: Optional split filter ('train', 'val', 'test').

        Yields:
            DatasetRecord: Parsed record.
        """
        file_path = self.resolve_dataset_file()

        if file_path.suffix.lower() == ".csv":
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    try:
                        record = parse_csv_row(row, record_id=idx)
                        if split is None or record.split == split:
                            yield record
                    except Exception as exc:
                        logger.warning("Skipping malformed row %d: %s", idx, exc)
        elif file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
            for idx, row in df.iterrows():
                try:
                    record_id = int(idx) if isinstance(idx, int) else str(idx)
                    record = parse_csv_row(dict(row), record_id=record_id)
                    if split is None or record.split == split:
                        yield record
                except Exception as exc:
                    logger.warning("Skipping malformed parquet row %s: %s", idx, exc)

    def load_records(
        self, split: str | None = None, limit: int | None = None
    ) -> list[DatasetRecord]:
        """Load all matching records into memory as a list.

        Args:
            split: Optional split filter ('train', 'val', 'test').
            limit: Optional maximum number of records to load.

        Returns:
            list[DatasetRecord]: List of parsed dataset records.
        """
        records: list[DatasetRecord] = []
        for record in self.stream_records(split=split):
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
        return records

    def load_split_dataframe(self, split: str | None = None) -> pd.DataFrame:
        """Load records as a pandas DataFrame."""
        records = self.load_records(split=split)
        rows: list[dict[str, Any]] = []
        for r in records:
            rows.append(
                {
                    "record_id": r.record_id,
                    "label": r.label,
                    "label_name": r.label_name,
                    "split": r.split,
                    "image": r.image,
                }
            )
        return pd.DataFrame(rows)

    def __iter__(self) -> Iterator[DatasetRecord]:
        """Iterate over all records in the dataset."""
        return self.stream_records()
