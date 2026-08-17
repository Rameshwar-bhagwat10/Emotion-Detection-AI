"""Unit tests for optimizer factory and parameter updates."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW

from ml.training.config import OptimizerConfig
from ml.training.optimizers import create_optimizer


def test_create_optimizer_adamw_default() -> None:
    """Verify default create_optimizer creates AdamW with correct hyperparameters."""
    model = nn.Linear(10, 2)
    config = OptimizerConfig(name="adamw", learning_rate=0.001, weight_decay=0.0001)
    optimizer = create_optimizer(model.parameters(), config)

    assert isinstance(optimizer, AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.001
    assert optimizer.param_groups[0]["weight_decay"] == 0.0001


def test_create_optimizer_from_dict() -> None:
    """Verify create_optimizer works with dictionary configuration."""
    model = nn.Linear(10, 2)
    cfg_dict = {"name": "adam", "learning_rate": 0.0005, "weight_decay": 0.0}
    optimizer = create_optimizer(model.parameters(), cfg_dict)

    assert isinstance(optimizer, Adam)
    assert optimizer.param_groups[0]["lr"] == 0.0005


def test_create_optimizer_sgd() -> None:
    """Verify SGD optimizer creation."""
    model = nn.Linear(10, 2)
    config = OptimizerConfig(name="sgd", learning_rate=0.01)
    optimizer = create_optimizer(model.parameters(), config)
    assert isinstance(optimizer, SGD)
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_optimizer_parameter_update_step() -> None:
    """Verify optimizer actually mutates parameters after backward pass."""
    model = nn.Linear(4, 2, bias=True)
    initial_weights = model.weight.clone()
    optimizer = create_optimizer(
        model.parameters(), OptimizerConfig(name="adamw", learning_rate=0.1)
    )

    inputs = torch.randn(2, 4)
    targets = torch.tensor([0, 1], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    assert not torch.equal(initial_weights, model.weight), "Optimizer failed to update weights"


def test_create_optimizer_invalid_name_raises() -> None:
    """Verify invalid optimizer name raises ValueError."""
    model = nn.Linear(10, 2)
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        create_optimizer(model.parameters(), {"name": "unknown_opt"})
