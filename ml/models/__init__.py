"""Package initialization for ML Models."""

from __future__ import annotations

from ml.models.cnn.baseline_cnn import BaselineCNN
from ml.models.cnn.blocks import ConvBlock
from ml.models.cnn.config import BaselineCNNConfig, StageConfig
from ml.models.factory import create_model
from ml.models.registry import (
    MODEL_REGISTRY,
    get_model_class,
    list_models,
    register_model,
)

__all__ = [
    "MODEL_REGISTRY",
    "BaselineCNN",
    "BaselineCNNConfig",
    "ConvBlock",
    "StageConfig",
    "create_model",
    "get_model_class",
    "list_models",
    "register_model",
]
