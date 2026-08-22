"""Configuration dataclasses for transfer learning models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TransferLearningConfig:
    """Configuration for transfer learning vision models."""

    model_name: str = "resnet18"
    num_classes: int = 7
    in_channels: int = 1
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.2
    use_attention: bool = False
    attention_type: str = "se"
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate transfer learning configuration parameters."""
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {self.dropout_rate}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransferLearningConfig:
        """Create configuration instance from dictionary."""
        valid_fields = {
            "model_name",
            "num_classes",
            "in_channels",
            "pretrained",
            "freeze_backbone",
            "dropout_rate",
            "use_attention",
            "attention_type",
            "extra_params",
        }
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> TransferLearningConfig:
        """Load configuration from a YAML file."""
        p = Path(yaml_path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
