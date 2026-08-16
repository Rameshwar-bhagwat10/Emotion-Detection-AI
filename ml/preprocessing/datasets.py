"""PyTorch Dataset implementation for FER2013 facial expression records.

Provides clean model-ready sample access, transform integration, and training metadata.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ml.datasets.fer2013.loader import FER2013Loader
from ml.datasets.fer2013.parser import DatasetRecord
from ml.preprocessing.transforms import (
    build_test_transform,
    build_train_transform,
    build_val_transform,
)


class FER2013Dataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset abstraction for FER2013 facial expression data."""

    def __init__(
        self,
        split: str = "train",
        data_path: str | Path = "data/raw/fer2013/fer2013.csv",
        records: Sequence[DatasetRecord] | None = None,
        transform: Callable[[Any], torch.Tensor] | None = None,
        target_transform: Callable[[int], int] | None = None,
        auto_transform: bool = True,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize FER2013Dataset.

        Args:
            split: Dataset split ('train', 'val', 'test').
            data_path: Path to raw FER2013 CSV or directory.
            records: Optional pre-loaded records (e.g. for synthetic fixtures/tests).
            transform: Optional explicit PyTorch transform.
            target_transform: Optional label transform.
            auto_transform: If True and transform is None, builds default transform for split.
            config: Optional configuration dictionary.
        """
        self.split = split.lower().strip()
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")

        self.data_path = Path(data_path)
        self.target_transform = target_transform

        # Load or assign records
        if records is not None:
            self.records: list[DatasetRecord] = [r for r in records if r.split == self.split]
        else:
            loader = FER2013Loader(self.data_path)
            self.records = loader.load_records(split=self.split)

        # Configure transform
        self.transform: Callable[[Any], torch.Tensor] | None
        if transform is not None:
            self.transform = transform
        elif auto_transform:
            if self.split == "train":
                self.transform = build_train_transform(config=config)
            elif self.split == "val":
                self.transform = build_val_transform(config=config)
            else:
                self.transform = build_test_transform(config=config)
        else:
            self.transform = None

    def __len__(self) -> int:
        """Return number of samples in the dataset split."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Retrieve a single processed, model-ready dataset sample.

        Args:
            index: Record index.

        Returns:
            dict containing 'image', 'label', 'emotion', 'record_id', 'split'.
        """
        record = self.records[index]
        image = record.image  # shape (48, 48), uint8

        if self.transform is not None:
            transformed_image = self.transform(image)
        else:
            # Fallback tensor conversion without normalization
            transformed_image = torch.from_numpy(image).unsqueeze(0).float() / 255.0

        label = record.label
        if self.target_transform is not None:
            label = self.target_transform(label)

        return {
            "image": transformed_image,
            "label": int(label),
            "emotion": record.label_name,
            "record_id": record.record_id,
            "split": record.split,
        }

    def get_class_counts(self) -> dict[str, int]:
        """Compute sample counts per emotion class for this dataset split."""
        counts: Counter[str] = Counter()
        for r in self.records:
            counts[r.label_name] += 1
        return dict(counts)

    def get_class_weights(self, num_classes: int = 7) -> torch.Tensor:
        """Calculate balanced inverse-frequency class weights for loss weighting.

        Formula: weight[c] = total_samples / (num_classes * count[c])

        Returns:
            torch.Tensor: Class weights tensor of shape [num_classes].
        """
        total = len(self.records)
        if total == 0:
            return torch.ones(num_classes, dtype=torch.float32)

        label_counts: Counter[int] = Counter()
        for r in self.records:
            label_counts[r.label] += 1

        weights = torch.zeros(num_classes, dtype=torch.float32)
        for class_idx in range(num_classes):
            c_count = label_counts.get(class_idx, 0)
            if c_count > 0:
                weights[class_idx] = total / (num_classes * c_count)
            else:
                weights[class_idx] = 1.0

        return weights
