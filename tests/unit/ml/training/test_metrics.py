"""Unit tests for metrics calculation and sample-weighted MetricTracker."""

from __future__ import annotations

import pytest
import torch

from ml.training.metrics import MetricTracker, calculate_accuracy


def test_calculate_accuracy_perfect() -> None:
    """Verify calculate_accuracy on 100% correct logits."""
    # Class 0, 1, 2
    logits = torch.tensor(
        [
            [10.0, 1.0, 0.0],
            [0.0, 10.0, 1.0],
            [0.0, 1.0, 10.0],
        ]
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    acc = calculate_accuracy(logits, labels)
    assert acc == 1.0


def test_calculate_accuracy_partial() -> None:
    """Verify calculate_accuracy with 50% accuracy."""
    logits = torch.tensor(
        [
            [10.0, 0.0],
            [10.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 1], dtype=torch.long)
    acc = calculate_accuracy(logits, labels)
    assert acc == 0.5


def test_calculate_accuracy_shape_validation() -> None:
    """Verify calculate_accuracy dimension checking."""
    with pytest.raises(ValueError, match="logits must be 2D"):
        calculate_accuracy(torch.randn(4), torch.tensor([0, 1, 2, 3]))

    with pytest.raises(ValueError, match="labels must be 1D"):
        calculate_accuracy(torch.randn(4, 2), torch.tensor([[0], [1], [0], [1]]))

    with pytest.raises(ValueError, match="Batch size mismatch"):
        calculate_accuracy(torch.randn(4, 2), torch.tensor([0, 1]))


def test_metric_tracker_sample_weighting() -> None:
    """Verify MetricTracker weights batches proportionally to batch size."""
    tracker = MetricTracker()

    # Batch 1: size 10, loss 2.0, all correct
    logits_1 = torch.zeros(10, 2)
    logits_1[:, 0] = 10.0
    labels_1 = torch.zeros(10, dtype=torch.long)
    tracker.update(loss_val=2.0, logits=logits_1, labels=labels_1, batch_size=10)

    # Batch 2: size 20, loss 1.0, none correct
    logits_2 = torch.zeros(20, 2)
    logits_2[:, 0] = 10.0
    labels_2 = torch.ones(20, dtype=torch.long)
    tracker.update(loss_val=1.0, logits=logits_2, labels=labels_2, batch_size=20)

    metrics = tracker.compute()
    # Expected weighted loss: (2.0 * 10 + 1.0 * 20) / 30 = 40 / 30 = 1.333333...
    assert pytest.approx(metrics["loss"], rel=1e-4) == 40.0 / 30.0
    # Expected weighted accuracy: (10 * 1.0 + 20 * 0.0) / 30 = 10 / 30 = 0.333333...
    assert pytest.approx(metrics["accuracy"], rel=1e-4) == 10.0 / 30.0

    tracker.reset()
    assert tracker.compute() == {"loss": 0.0, "accuracy": 0.0}
