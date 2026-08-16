"""Integration tests for Phase 03 Preprocessing Pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml.preprocessing.dataloaders import build_dataloaders
from ml.preprocessing.datasets import FER2013Dataset
from ml.preprocessing.normalization import DEFAULT_TRAIN_MEAN, DEFAULT_TRAIN_STD, denormalize

RAW_CSV_PATH = Path("data/raw/fer2013/fer2013.csv")


def test_real_dataset_preprocessing_cardinality_and_batches() -> None:
    """MANDATORY: Verify full preprocessing pipeline on the actual FER2013 dataset."""
    if not RAW_CSV_PATH.exists():
        pytest.skip("Raw FER2013 dataset not downloaded yet.")

    train_ds = FER2013Dataset(split="train", data_path=RAW_CSV_PATH)
    val_ds = FER2013Dataset(split="val", data_path=RAW_CSV_PATH)
    test_ds = FER2013Dataset(split="test", data_path=RAW_CSV_PATH)

    # 1. Cardinality check matching Phase 02 exact values
    assert len(train_ds) == 28709
    assert len(val_ds) == 3589
    assert len(test_ds) == 3589
    assert len(train_ds) + len(val_ds) + len(test_ds) == 35887

    # 2. Sample contract check
    train_sample = train_ds[0]
    assert train_sample["image"].shape == (1, 48, 48)
    assert train_sample["image"].dtype == torch.float32
    assert 0 <= train_sample["label"] <= 6

    # 3. DataLoaders batch generation
    train_loader, val_loader, test_loader = build_dataloaders(data_path=RAW_CSV_PATH, batch_size=32)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    assert train_batch["image"].shape == (32, 1, 48, 48)
    assert train_batch["image"].dtype == torch.float32
    assert train_batch["label"].shape == (32,)
    assert val_batch["image"].shape == (32, 1, 48, 48)
    assert test_batch["image"].shape == (32, 1, 48, 48)

    # 4. Denormalize check
    denorm_sample = denormalize(
        train_sample["image"], mean=DEFAULT_TRAIN_MEAN, std=DEFAULT_TRAIN_STD
    )
    assert denorm_sample.min() >= 0.0
    assert denorm_sample.max() <= 1.0
