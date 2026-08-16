"""Unit tests for FER2013 statistical analyzer and duplicate detector."""

import numpy as np

from ml.datasets.fer2013.parser import DatasetRecord
from ml.datasets.fer2013.statistics import FER2013Statistics


def test_statistics_computation_synthetic() -> None:
    """Verify statistical metrics on deterministic synthetic records."""
    analyzer = FER2013Statistics()

    # 4 records: 2 happy, 1 angry, 1 neutral
    # Images with known constant values: 0, 100, 200, 255
    r1 = DatasetRecord(1, np.full((48, 48), 0, dtype=np.uint8), 3, "happy", "train")
    r2 = DatasetRecord(2, np.full((48, 48), 100, dtype=np.uint8), 3, "happy", "val")
    r3 = DatasetRecord(3, np.full((48, 48), 200, dtype=np.uint8), 0, "angry", "train")
    r4 = DatasetRecord(4, np.full((48, 48), 255, dtype=np.uint8), 6, "neutral", "test")

    stats = analyzer.compute_statistics([r1, r2, r3, r4])

    assert stats.total_samples == 4
    assert stats.num_classes == 7
    assert stats.classes["happy"] == 2
    assert stats.classes["angry"] == 1
    assert stats.classes["neutral"] == 1
    assert stats.classes["disgust"] == 0

    assert stats.splits["train"] == 2
    assert stats.splits["val"] == 1
    assert stats.splits["test"] == 1

    assert stats.image_dimensions["min_width"] == 48
    assert stats.image_dimensions["max_width"] == 48
    assert stats.image_dimensions["uniform_48x48"] is True

    assert stats.pixel_statistics["min"] == 0.0
    assert stats.pixel_statistics["max"] == 255.0


def test_duplicate_and_leakage_detection_synthetic() -> None:
    """Verify duplicate detection and cross-split leakage calculation on synthetic data."""
    analyzer = FER2013Statistics()

    # Identical image arrA in train and val (cross-split leakage)
    arr_a = np.full((48, 48), 50, dtype=np.uint8)
    # Identical image arrB duplicated twice in train (internal duplicate)
    arr_b = np.full((48, 48), 150, dtype=np.uint8)
    # Unique image arrC in test
    arr_c = np.full((48, 48), 220, dtype=np.uint8)

    r1 = DatasetRecord(1, arr_a, 0, "angry", "train")
    r2 = DatasetRecord(2, arr_a, 0, "angry", "val")
    r3 = DatasetRecord(3, arr_b, 3, "happy", "train")
    r4 = DatasetRecord(4, arr_b, 3, "happy", "train")
    r5 = DatasetRecord(5, arr_c, 5, "surprise", "test")

    dup_result = analyzer.compute_duplicates([r1, r2, r3, r4, r5])

    assert dup_result.total_samples == 5
    assert dup_result.unique_samples == 3
    assert dup_result.exact_duplicates_count == 2
    assert dup_result.duplicate_groups_count == 2
    assert dup_result.duplicates_within_train == 1
    assert dup_result.train_val_overlap == 1
    assert dup_result.cross_split_leakage_detected is True
