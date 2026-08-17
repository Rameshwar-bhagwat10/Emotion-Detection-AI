"""Unit tests for learning rate scheduler factory and step behavior."""

from __future__ import annotations

import pytest
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from ml.training.config import SchedulerConfig
from ml.training.schedulers import create_scheduler


def test_create_scheduler_reduce_on_plateau() -> None:
    """Verify ReduceLROnPlateau creation and metric dependency flag."""
    model = nn.Linear(10, 2)
    optimizer = AdamW(model.parameters(), lr=0.01)
    config = SchedulerConfig(
        name="reduce_on_plateau",
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    scheduler, is_metric_dep = create_scheduler(optimizer, config)

    assert isinstance(scheduler, ReduceLROnPlateau)
    assert is_metric_dep is True

    # Test plateau stepping
    initial_lr = optimizer.param_groups[0]["lr"]
    # Provide non-improving validation loss for patience + 1 epochs
    scheduler.step(10.0)
    scheduler.step(10.0)
    scheduler.step(10.0)
    scheduler.step(10.0)

    reduced_lr = optimizer.param_groups[0]["lr"]
    assert reduced_lr < initial_lr
    assert reduced_lr == initial_lr * 0.5


def test_create_scheduler_steplr() -> None:
    """Verify StepLR creation and step behavior."""
    model = nn.Linear(10, 2)
    optimizer = AdamW(model.parameters(), lr=0.01)
    config = SchedulerConfig(name="step_lr", step_size=2, gamma=0.5)
    scheduler, is_metric_dep = create_scheduler(optimizer, config)

    assert isinstance(scheduler, StepLR)
    assert is_metric_dep is False

    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.01
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.005


def test_create_scheduler_cosine_annealing() -> None:
    """Verify CosineAnnealingLR creation."""
    model = nn.Linear(10, 2)
    optimizer = AdamW(model.parameters(), lr=0.01)
    config = SchedulerConfig(name="cosine_annealing", t_max=10, min_lr=1e-5)
    scheduler, is_metric_dep = create_scheduler(optimizer, config)

    assert isinstance(scheduler, CosineAnnealingLR)
    assert is_metric_dep is False


def test_create_scheduler_invalid_name_raises() -> None:
    """Verify invalid scheduler name raises ValueError."""
    model = nn.Linear(10, 2)
    optimizer = AdamW(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        create_scheduler(optimizer, {"name": "invalid_scheduler"})
