"""Unit tests for Phase 03 DataLoaders."""

from __future__ import annotations

import numpy as np
import torch

from ml.datasets.fer2013.parser import DatasetRecord
from ml.preprocessing.dataloaders import (
    create_test_loader,
    create_train_loader,
    create_val_loader,
)
from ml.preprocessing.datasets import FER2013Dataset


def _make_dummy_dataset(size: int = 16, split: str = "train") -> FER2013Dataset:
    """Helper to create synthetic FER2013Dataset."""
    records: list[DatasetRecord] = []
    for i in range(size):
        records.append(
            DatasetRecord(
                record_id=f"{split}_{i}",
                image=np.full((48, 48), i * 10, dtype=np.uint8),
                label=i % 7,
                label_name=["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"][
                    i % 7
                ],
                split=split,
            )
        )
    return FER2013Dataset(split=split, records=records)


def test_train_dataloader_batch_shapes_and_shuffling() -> None:
    """MANDATORY: Verify training DataLoader batch tensor dimensions and properties."""
    ds = _make_dummy_dataset(size=16, split="train")
    loader = create_train_loader(dataset=ds, batch_size=4, shuffle=True)

    batch = next(iter(loader))

    assert "image" in batch
    assert "label" in batch
    assert batch["image"].shape == (4, 1, 48, 48)
    assert batch["image"].dtype == torch.float32
    assert batch["label"].shape == (4,)
    assert batch["label"].dtype == torch.int64


def test_val_and_test_dataloader_non_shuffling() -> None:
    """Verify validation and test DataLoaders do not shuffle."""
    val_ds = _make_dummy_dataset(size=8, split="val")
    val_loader = create_val_loader(dataset=val_ds, batch_size=4)

    test_ds = _make_dummy_dataset(size=8, split="test")
    test_loader = create_test_loader(dataset=test_ds, batch_size=4)

    # First batch labels should be strictly in sequential order 0, 1, 2, 3
    val_batch = next(iter(val_loader))
    assert val_batch["label"].tolist() == [0, 1, 2, 3]

    test_batch = next(iter(test_loader))
    assert test_batch["label"].tolist() == [0, 1, 2, 3]
