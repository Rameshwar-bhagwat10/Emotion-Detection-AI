"""Unit tests for Model Factory and Model Registry."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ml.models.cnn.baseline_cnn import BaselineCNN
from ml.models.factory import create_model
from ml.models.registry import get_model_class, list_models, register_model


def test_create_model_baseline_cnn() -> None:
    """Verify create_model instantiates BaselineCNN with default configuration."""
    model = create_model("baseline_cnn")

    assert isinstance(model, BaselineCNN)
    assert model.input_channels == 1
    assert model.num_classes == 7


def test_create_model_unknown_raises() -> None:
    """Verify create_model raises ValueError for unregistered models."""
    with pytest.raises(ValueError, match="Unknown model 'non_existent_architecture'"):
        create_model("non_existent_architecture")


def test_model_registry_custom_registration() -> None:
    """Verify custom models can be registered and retrieved dynamically."""

    @register_model("dummy_test_model")
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    assert "dummy_test_model" in list_models()
    cls = get_model_class("dummy_test_model")
    assert cls is DummyModel

    inst = create_model("dummy_test_model")
    assert isinstance(inst, DummyModel)
