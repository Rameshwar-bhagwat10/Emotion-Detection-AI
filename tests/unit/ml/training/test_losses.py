"""Unit tests for loss function creation and computation."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ml.training.config import LossConfig
from ml.training.losses import create_loss


def test_create_loss_cross_entropy_default() -> None:
    """Verify default create_loss returns unweighted nn.CrossEntropyLoss."""
    criterion = create_loss("cross_entropy")
    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.weight is None


def test_create_loss_from_config() -> None:
    """Verify create_loss accepts LossConfig dataclass."""
    config = LossConfig(name="cross_entropy", class_weighted=False)
    criterion = create_loss(config)
    assert isinstance(criterion, nn.CrossEntropyLoss)


def test_create_loss_with_class_weights() -> None:
    """Verify create_loss applies class weights when class_weighted is True."""
    weights = torch.tensor([1.0, 2.0, 1.0, 0.5, 1.0, 1.5, 1.0])
    config = LossConfig(name="cross_entropy", class_weighted=True)
    criterion = create_loss(config, class_weights=weights)
    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.weight is not None
    assert torch.equal(criterion.weight, weights)


def test_create_loss_forward_and_backward() -> None:
    """Verify cross entropy forward and backward pass on raw logits."""
    criterion = create_loss("cross_entropy")
    logits = torch.randn(4, 7, requires_grad=True)
    labels = torch.tensor([0, 3, 6, 2], dtype=torch.long)

    loss = criterion(logits, labels)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_create_loss_invalid_name_raises() -> None:
    """Verify invalid loss name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported loss function"):
        create_loss("invalid_loss")
