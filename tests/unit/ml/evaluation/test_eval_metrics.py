"""Unit tests for evaluation metrics computation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.evaluation.metrics import (
    calculate_accuracy,
    calculate_macro_metrics,
    calculate_per_class_metrics,
    calculate_weighted_metrics,
)


def test_calculate_accuracy_perfect() -> None:
    """Verify calculate_accuracy on 100% accurate predictions."""
    y_true = [0, 1, 2, 3, 4, 5, 6]
    y_pred = [0, 1, 2, 3, 4, 5, 6]
    acc = calculate_accuracy(y_true, y_pred)
    assert acc == 1.0


def test_calculate_accuracy_mismatch() -> None:
    """Verify calculate_accuracy on partial accuracy."""
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 0, 0])
    acc = calculate_accuracy(y_true, y_pred)
    assert acc == 0.5


def test_calculate_accuracy_tensor_input() -> None:
    """Verify calculate_accuracy accepts PyTorch tensors."""
    y_true = torch.tensor([1, 2, 3])
    y_pred = torch.tensor([1, 0, 3])
    acc = calculate_accuracy(y_true, y_pred)
    assert pytest.approx(acc, rel=1e-4) == 2.0 / 3.0


def test_calculate_accuracy_dimension_mismatch_raises() -> None:
    """Verify error on length mismatch."""
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_accuracy([0, 1], [0, 1, 2])


def test_calculate_per_class_metrics_known_values() -> None:
    """Verify exact precision, recall, and F1 values on known synthetic predictions."""
    class_names = ["angry", "happy", "sad"]
    y_true = [0, 0, 1, 1, 2, 2]
    # Class 0: true=[0,0], pred=[0,1] -> TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5, F1=0.6667
    # Class 1: true=[1,1], pred=[1,1] + FP from 0 -> TP=2, FP=1, FN=0 -> Prec=2/3=0.6667, Rec=1.0, F1=0.8
    # Class 2: true=[2,2], pred=[2,2] -> TP=2, FP=0, FN=0 -> Prec=1.0, Rec=1.0, F1=1.0
    y_pred = [0, 1, 1, 1, 2, 2]

    per_class = calculate_per_class_metrics(y_true, y_pred, num_classes=3, class_names=class_names)

    assert per_class["angry"]["precision"] == 1.0
    assert per_class["angry"]["recall"] == 0.5
    assert pytest.approx(per_class["angry"]["f1"], rel=1e-3) == 2.0 / 3.0
    assert per_class["angry"]["support"] == 2

    assert pytest.approx(per_class["happy"]["precision"], rel=1e-3) == 2.0 / 3.0
    assert per_class["happy"]["recall"] == 1.0
    assert pytest.approx(per_class["happy"]["f1"], rel=1e-3) == 0.8
    assert per_class["happy"]["support"] == 2

    assert per_class["sad"]["precision"] == 1.0
    assert per_class["sad"]["recall"] == 1.0
    assert per_class["sad"]["f1"] == 1.0
    assert per_class["sad"]["support"] == 2


def test_calculate_macro_and_weighted_metrics() -> None:
    """Verify macro and weighted metric calculations."""
    per_class = {
        "c0": {"precision": 1.0, "recall": 0.5, "f1": 0.666667, "support": 10},
        "c1": {"precision": 0.5, "recall": 1.0, "f1": 0.666667, "support": 20},
    }

    macro = calculate_macro_metrics(per_class)
    assert macro["macro_precision"] == 0.75
    assert macro["macro_recall"] == 0.75
    assert pytest.approx(macro["macro_f1"], rel=1e-4) == 0.666667

    weighted = calculate_weighted_metrics(per_class)
    # Expected weighted precision: (1.0 * 10 + 0.5 * 20) / 30 = 20 / 30 = 0.666667
    assert pytest.approx(weighted["weighted_precision"], rel=1e-4) == 20.0 / 30.0
    # Expected weighted recall: (0.5 * 10 + 1.0 * 20) / 30 = 25 / 30 = 0.833333
    assert pytest.approx(weighted["weighted_recall"], rel=1e-4) == 25.0 / 30.0


def test_zero_division_safety() -> None:
    """Verify zero division handled safely when class is never predicted."""
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]
    per_class = calculate_per_class_metrics(y_true, y_pred, num_classes=2, class_names=["a", "b"])

    assert per_class["b"]["precision"] == 0.0
    assert per_class["b"]["recall"] == 0.0
    assert per_class["b"]["f1"] == 0.0
    assert per_class["b"]["support"] == 0
