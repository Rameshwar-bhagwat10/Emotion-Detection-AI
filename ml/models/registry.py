"""Model Registry for registering and retrieving model architecture classes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from torch import nn

T = TypeVar("T", bound=type[nn.Module])

MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(name: str | None = None) -> Callable[[T], T]:
    """Decorator to register a model architecture class in MODEL_REGISTRY.

    Args:
        name: Unique string identifier for the model. If None, uses class.__name__.lower().

    Returns:
        Decorator function.
    """

    def decorator(cls: T) -> T:
        model_name = (name or cls.__name__).lower().strip()
        if model_name in MODEL_REGISTRY:
            # Overwrite or maintain
            pass
        MODEL_REGISTRY[model_name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> type[nn.Module]:
    """Retrieve registered model architecture class by name.

    Args:
        name: Name of the model (e.g. 'baseline_cnn').

    Returns:
        Model class.

    Raises:
        ValueError: If model name is not found in the registry.
    """
    clean_name = name.lower().strip()
    if clean_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys())) or "none"
        raise ValueError(f"Unknown model '{name}'. Available registered models: [{available}]")
    return MODEL_REGISTRY[clean_name]


def list_models() -> list[str]:
    """Return a list of all currently registered model names."""
    return sorted(MODEL_REGISTRY.keys())
