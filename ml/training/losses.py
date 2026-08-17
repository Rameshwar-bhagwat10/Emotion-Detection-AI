"""Loss function factory and utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ml.training.config import LossConfig


def create_loss(
    config: LossConfig | dict[str, Any] | str,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """Create and return a configured loss function.

    Args:
        config: LossConfig instance, dictionary, or string name.
        class_weights: Optional 1D tensor of class weights for CrossEntropyLoss.

    Returns:
        Configured PyTorch nn.Module loss criterion.
    """
    if isinstance(config, str):
        loss_name = config.lower().strip()
        class_weighted = False
    elif isinstance(config, dict):
        loss_name = str(config.get("name", "cross_entropy")).lower().strip()
        class_weighted = bool(config.get("class_weighted", False))
    elif isinstance(config, LossConfig):
        loss_name = config.name.lower().strip()
        class_weighted = config.class_weighted
    else:
        raise TypeError(f"Unsupported config type for create_loss: {type(config)}")

    if loss_name in ("cross_entropy", "crossentropy", "ce"):
        weights = class_weights if class_weighted else None
        return nn.CrossEntropyLoss(weight=weights)

    raise ValueError(
        f"Unsupported loss function '{loss_name}'. Supported losses: ['cross_entropy']"
    )
