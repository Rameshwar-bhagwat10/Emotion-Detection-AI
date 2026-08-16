"""Unit tests for Phase 03 Normalization and Data Leakage Prevention."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.datasets.fer2013.parser import DatasetRecord
from ml.preprocessing.normalization import (
    DEFAULT_TRAIN_MEAN,
    DEFAULT_TRAIN_STD,
    NormalizationStats,
    calculate_train_normalization_stats,
    denormalize,
    normalize,
)


def _make_dummy_record(val: int, split: str, record_id: int = 0) -> DatasetRecord:
    """Helper to create uniform synthetic image records."""
    img = np.full((48, 48), val, dtype=np.uint8)
    return DatasetRecord(
        record_id=record_id,
        image=img,
        label=0,
        label_name="angry",
        split=split,
    )


def test_calculate_train_normalization_stats_synthetic() -> None:
    """Verify exact mean and std calculation on controlled synthetic arrays."""
    # 2 records with pixel values 0 and 255 -> scaled values 0.0 and 1.0
    r1 = _make_dummy_record(0, split="train", record_id=1)
    r2 = _make_dummy_record(255, split="train", record_id=2)

    stats = calculate_train_normalization_stats([r1, r2], split="train")

    assert isinstance(stats, NormalizationStats)
    assert stats.total_samples == 2
    assert stats.split == "train"
    assert pytest.approx(stats.mean, rel=1e-4) == 0.5
    assert pytest.approx(stats.std, rel=1e-4) == 0.5
    assert pytest.approx(stats.raw_mean, rel=1e-4) == 127.5
    assert pytest.approx(stats.raw_std, rel=1e-4) == 127.5


def test_calculate_train_normalization_stats_empty_split() -> None:
    """Verify ValueError is raised if the target split contains zero records."""
    r1 = _make_dummy_record(128, split="val", record_id=1)
    with pytest.raises(ValueError, match="No records found for split 'train'"):
        calculate_train_normalization_stats([r1], split="train")


def test_normalize_2d_3d_4d_shapes() -> None:
    """Verify normalization broadcasts accurately across 2D, 3D, and 4D tensors."""
    mean = 0.5
    std = 0.25

    # 1. 2D [48, 48]
    t2 = torch.full((48, 48), 0.75, dtype=torch.float32)
    norm2 = normalize(t2, mean=mean, std=std)
    assert norm2.shape == (48, 48)
    assert pytest.approx(norm2[0, 0].item(), abs=1e-5) == (0.75 - 0.5) / 0.25  # 1.0

    # 2. 3D [1, 48, 48]
    t3 = torch.full((1, 48, 48), 0.5, dtype=torch.float32)
    norm3 = normalize(t3, mean=mean, std=std)
    assert norm3.shape == (1, 48, 48)
    assert pytest.approx(norm3[0, 0, 0].item(), abs=1e-5) == 0.0

    # 3. 4D [4, 1, 48, 48]
    t4 = torch.full((4, 1, 48, 48), 0.0, dtype=torch.float32)
    norm4 = normalize(t4, mean=mean, std=std)
    assert norm4.shape == (4, 1, 48, 48)
    assert pytest.approx(norm4[0, 0, 0, 0].item(), abs=1e-5) == (0.0 - 0.5) / 0.25  # -2.0


def test_normalize_invalid_std() -> None:
    """Verify ValueError is raised when standard deviation is <= 0."""
    t = torch.zeros((1, 48, 48), dtype=torch.float32)
    with pytest.raises(ValueError, match="strictly positive"):
        normalize(t, mean=0.5, std=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        normalize(t, mean=0.5, std=-0.1)


def test_denormalize_invertibility() -> None:
    """Verify denormalize(normalize(x)) reconstructs original tensor values."""
    original = torch.tensor([[[0.0, 0.25], [0.5, 1.0]]], dtype=torch.float32)
    norm = normalize(original, mean=DEFAULT_TRAIN_MEAN, std=DEFAULT_TRAIN_STD)
    denorm = denormalize(norm, mean=DEFAULT_TRAIN_MEAN, std=DEFAULT_TRAIN_STD)

    assert torch.allclose(original, denorm, atol=1e-5)


def test_normalization_leakage_isolation() -> None:
    """MANDATORY: Verify that validation and test pixels do NOT leak into TRAIN statistics."""
    # Synthetic dataset with deliberately divergent pixel distributions per split
    train_records = [_make_dummy_record(100, split="train", record_id=i) for i in range(10)]
    val_records = [_make_dummy_record(250, split="val", record_id=10 + i) for i in range(10)]
    test_records = [_make_dummy_record(10, split="test", record_id=20 + i) for i in range(10)]

    all_records = train_records + val_records + test_records

    # Compute stats using train split only
    stats = calculate_train_normalization_stats(all_records, split="train")

    expected_train_scaled = 100.0 / 255.0
    assert pytest.approx(stats.mean, rel=1e-4) == expected_train_scaled
    assert pytest.approx(stats.std, abs=1e-5) == 0.0
    assert pytest.approx(stats.raw_mean, rel=1e-4) == 100.0
    assert stats.total_samples == 10

    # Ensure validation (250) and test (10) had zero mathematical influence
    assert stats.raw_mean != pytest.approx(250.0)
    assert stats.raw_mean != pytest.approx(10.0)
    assert stats.raw_mean != pytest.approx((100 * 10 + 250 * 10 + 10 * 10) / 30.0)
