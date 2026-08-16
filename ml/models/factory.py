"""Model Factory for instantiating model architectures from configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from ml.models.cnn.baseline_cnn import BaselineCNN
from ml.models.cnn.config import BaselineCNNConfig
from ml.models.registry import get_model_class, register_model

# Explicitly register known architectures
register_model("baseline_cnn")(BaselineCNN)
register_model("baseline_cnn_v1")(BaselineCNN)


def create_model(
    model_name: str = "baseline_cnn",
    config: BaselineCNNConfig | dict[str, Any] | None = None,
    config_path: str | Path = "ml/configs/models.yaml",
    **kwargs: Any,
) -> nn.Module:
    """Create and initialize a model instance based on name and configuration.

    Args:
        model_name: Identifier of the model to create (e.g. 'baseline_cnn').
        config: Optional configuration object or dictionary.
        config_path: Path to models.yaml configuration file if config is not provided.
        **kwargs: Overrides for configuration arguments.

    Returns:
        Instantiated PyTorch nn.Module.

    Raises:
        ValueError: If model_name is not registered.
    """
    model_cls = get_model_class(model_name)

    if config is None:
        if model_cls is BaselineCNN:
            p = Path(config_path)
            if p.exists():
                cfg = BaselineCNNConfig.from_yaml(p)
            else:
                cfg = BaselineCNNConfig(**kwargs)
            return BaselineCNN(config=cfg, **kwargs)
        return model_cls(**kwargs)

    if isinstance(config, BaselineCNNConfig):
        return model_cls(config=config, **kwargs)

    if isinstance(config, dict):
        if model_cls is BaselineCNN:
            cfg = BaselineCNNConfig.from_dict(config)
            return BaselineCNN(config=cfg, **kwargs)
        return model_cls(config=config, **kwargs)

    return model_cls(config=config, **kwargs)
