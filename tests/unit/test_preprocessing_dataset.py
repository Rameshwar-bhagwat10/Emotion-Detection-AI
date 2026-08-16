"""Unit tests for Phase 03 FER2013Dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.datasets.fer2013.parser import DatasetRecord
from ml.preprocessing.datasets import FER2013Dataset


def _make_dummy_records() -> list[DatasetRecord]:
    """Generate controlled dummy records across train, val, and test."""
    records: list[DatasetRecord] = []
    # 5 train records with classes 0..4
    for i in range(5):
        records.append(
            DatasetRecord(
                record_id=f"train_{i}",
                image=np.full((48, 48), 50 * i, dtype=np.uint8),
                label=i,
                label_name=["angry", "disgust", "fear", "happy", "sad"][i],
                split="train",
            )
        )
    # 2 val records with classes 3, 4
    for i in range(2):
        records.append(
            DatasetRecord(
                record_id=f"val_{i}",
                image=np.full((48, 48), 100, dtype=np.uint8),
                label=i + 3,
                label_name=["happy", "sad"][i],
                split="val",
            )
        )
    # 2 test records
    for i in range(2):
        records.append(
            DatasetRecord(
                record_id=f"test_{i}",
                image=np.full((48, 48), 200, dtype=np.uint8),
                label=i,
                label_name=["angry", "disgust"][i],
                split="test",
            )
        )
    return records


def test_fer2013_dataset_split_filtering_and_len() -> None:
    """Verify FER2013Dataset correctly filters by split and reports accurate length."""
    records = _make_dummy_records()

    train_ds = FER2013Dataset(split="train", records=records)
    val_ds = FER2013Dataset(split="val", records=records)
    test_ds = FER2013Dataset(split="test", records=records)

    assert len(train_ds) == 5
    assert len(val_ds) == 2
    assert len(test_ds) == 2


def test_fer2013_dataset_invalid_split_raises() -> None:
    """Verify invalid split names raise ValueError."""
    records = _make_dummy_records()
    with pytest.raises(ValueError, match="Invalid split"):
        FER2013Dataset(split="invalid_split", records=records)


def test_fer2013_dataset_item_contract_and_label_integrity() -> None:
    """MANDATORY: Verify sample dictionary keys, tensor shapes, and label integrity."""
    records = _make_dummy_records()
    train_ds = FER2013Dataset(split="train", records=records)

    sample = train_ds[0]

    assert "image" in sample
    assert "label" in sample
    assert "emotion" in sample
    assert "record_id" in sample
    assert "split" in sample

    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (1, 48, 48)
    assert sample["image"].dtype == torch.float32
    assert sample["label"] == 0
    assert sample["emotion"] == "angry"
    assert sample["record_id"] == "train_0"
    assert sample["split"] == "train"


def test_fer2013_dataset_class_weights_calculation() -> None:
    """Verify class weights calculation computes balanced inverse frequencies."""
    records = _make_dummy_records()
    train_ds = FER2013Dataset(split="train", records=records)

    weights = train_ds.get_class_weights(num_classes=7)

    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (7,)
    assert (weights > 0).all()
    # Classes 0..4 each have count=1 out of 5 samples: weight = 5 / (7 * 1) = 5/7 = 0.714
    for c in range(5):
        assert pytest.approx(weights[c].item(), rel=1e-3) == 5.0 / 7.0
