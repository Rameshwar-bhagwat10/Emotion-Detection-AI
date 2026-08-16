"""Configuration models and schema validation for Baseline CNN architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StageConfig:
    """Configuration for a single convolutional stage."""

    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    pool: bool = True
    pool_kernel: int = 2
    pool_stride: int = 2

    def __post_init__(self) -> None:
        """Validate stage parameters."""
        if self.in_channels <= 0:
            raise ValueError(f"Stage in_channels must be > 0, got {self.in_channels}")
        if self.out_channels <= 0:
            raise ValueError(f"Stage out_channels must be > 0, got {self.out_channels}")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(
                f"Stage kernel_size must be positive odd integer, got {self.kernel_size}"
            )
        if self.stride <= 0:
            raise ValueError(f"Stage stride must be > 0, got {self.stride}")
        if self.padding < 0:
            raise ValueError(f"Stage padding must be >= 0, got {self.padding}")
        if self.pool and (self.pool_kernel <= 0 or self.pool_stride <= 0):
            raise ValueError(
                f"Pool kernel and stride must be > 0, got kernel={self.pool_kernel}, stride={self.pool_stride}"
            )


@dataclass
class BaselineCNNConfig:
    """Validated configuration for BaselineCNN model."""

    name: str = "baseline_cnn"
    version: str = "baseline_cnn_v1"
    description: str = "4-stage Convolutional Neural Network for FER"
    input_channels: int = 1
    input_size: tuple[int, int] = (48, 48)
    num_classes: int = 7
    stages: list[StageConfig] = field(
        default_factory=lambda: [
            StageConfig(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, pool=True
            ),
            StageConfig(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, pool=True
            ),
            StageConfig(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, pool=True
            ),
            StageConfig(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, pool=True
            ),
        ]
    )
    adaptive_pool_size: tuple[int, int] = (1, 1)
    classifier_hidden: int = 128
    dropout: float = 0.3
    activation: str = "relu"
    initialization: str = "kaiming_normal"

    def __post_init__(self) -> None:
        """Validate entire model configuration parameters."""
        if self.input_channels <= 0:
            raise ValueError(f"input_channels must be > 0, got {self.input_channels}")
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {self.num_classes}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in range [0.0, 1.0), got {self.dropout}")
        if self.classifier_hidden <= 0:
            raise ValueError(f"classifier_hidden must be > 0, got {self.classifier_hidden}")
        if not self.stages:
            raise ValueError("stages cannot be empty")
        if self.stages[0].in_channels != self.input_channels:
            raise ValueError(
                f"First stage in_channels ({self.stages[0].in_channels}) must match model input_channels ({self.input_channels})"
            )
        # Check channel connectivity across stages
        for i in range(len(self.stages) - 1):
            if self.stages[i].out_channels != self.stages[i + 1].in_channels:
                raise ValueError(
                    f"Stage {i} out_channels ({self.stages[i].out_channels}) does not match Stage {i + 1} in_channels ({self.stages[i + 1].in_channels})"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineCNNConfig:
        """Construct and validate BaselineCNNConfig from dictionary."""
        model_data = data.get("models", {}).get("baseline_cnn", data.get("baseline_cnn", data))

        raw_stages = model_data.get("stages", [])
        stages: list[StageConfig] = []
        if raw_stages:
            for s in raw_stages:
                if isinstance(s, StageConfig):
                    stages.append(s)
                else:
                    stages.append(
                        StageConfig(
                            in_channels=int(s["in_channels"]),
                            out_channels=int(s["out_channels"]),
                            kernel_size=int(s.get("kernel_size", 3)),
                            stride=int(s.get("stride", 1)),
                            padding=int(s.get("padding", 1)),
                            pool=bool(s.get("pool", True)),
                            pool_kernel=int(s.get("pool_kernel", 2)),
                            pool_stride=int(s.get("pool_stride", 2)),
                        )
                    )

        input_size_raw = model_data.get("input_size", (48, 48))
        input_size = (
            tuple(input_size_raw) if isinstance(input_size_raw, (list, tuple)) else (48, 48)
        )

        pool_size_raw = model_data.get("adaptive_pool_size", (1, 1))
        adaptive_pool_size = (
            tuple(pool_size_raw) if isinstance(pool_size_raw, (list, tuple)) else (1, 1)
        )

        return cls(
            name=str(model_data.get("name", "baseline_cnn")),
            version=str(model_data.get("version", "baseline_cnn_v1")),
            description=str(
                model_data.get("description", "4-stage Convolutional Neural Network for FER")
            ),
            input_channels=int(model_data.get("input_channels", 1)),
            input_size=input_size,  # type: ignore[arg-type]
            num_classes=int(model_data.get("num_classes", 7)),
            stages=stages if stages else cls().stages,
            adaptive_pool_size=adaptive_pool_size,  # type: ignore[arg-type]
            classifier_hidden=int(model_data.get("classifier_hidden", 128)),
            dropout=float(model_data.get("dropout", 0.3)),
            activation=str(model_data.get("activation", "relu")),
            initialization=str(model_data.get("initialization", "kaiming_normal")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path = "ml/configs/models.yaml") -> BaselineCNNConfig:
        """Load and parse BaselineCNNConfig from YAML file."""
        cfg_path = Path(path)
        if not cfg_path.exists():
            return cls()
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})
