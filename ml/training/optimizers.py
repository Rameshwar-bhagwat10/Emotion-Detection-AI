"""Optimizer factory and parameter group utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import SGD, Adam, AdamW, Optimizer

from ml.training.config import OptimizerConfig


def create_optimizer(
    params: Iterable[torch.nn.Parameter] | list[dict[str, Any]],
    config: OptimizerConfig | dict[str, Any],
) -> Optimizer:
    """Create and return a configured PyTorch optimizer.

    Args:
        params: Model parameters iterable or parameter group list.
        config: OptimizerConfig instance or configuration dictionary.

    Returns:
        Configured PyTorch Optimizer instance.
    """
    if isinstance(config, dict):
        betas_val = config.get("betas", [0.9, 0.999])
        opt_config = OptimizerConfig(
            name=str(config.get("name", "adamw")).lower().strip(),
            learning_rate=float(config.get("learning_rate", 0.001)),
            weight_decay=float(config.get("weight_decay", 0.0001)),
            betas=(float(betas_val[0]), float(betas_val[1])),
            eps=float(config.get("eps", 1e-8)),
        )
    elif isinstance(config, OptimizerConfig):
        opt_config = config
    else:
        raise TypeError(f"Unsupported config type for create_optimizer: {type(config)}")

    name = opt_config.name.lower().strip()

    if name == "adamw":
        return AdamW(
            params=params,
            lr=opt_config.learning_rate,
            weight_decay=opt_config.weight_decay,
            betas=opt_config.betas,
            eps=opt_config.eps,
        )
    elif name == "adam":
        return Adam(
            params=params,
            lr=opt_config.learning_rate,
            weight_decay=opt_config.weight_decay,
            betas=opt_config.betas,
            eps=opt_config.eps,
        )
    elif name == "sgd":
        return SGD(
            params=params,
            lr=opt_config.learning_rate,
            weight_decay=opt_config.weight_decay,
            momentum=0.9,
        )

    raise ValueError(
        f"Unsupported optimizer '{name}'. Supported optimizers: ['adamw', 'adam', 'sgd']"
    )
