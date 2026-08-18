"""Configuration schemas and dataclasses for model optimization experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FP16Config:
    """Configuration for FP16 reduced precision evaluation."""

    enabled: bool = True
    device: str = "cpu"


@dataclass
class QuantizationConfig:
    """Configuration for Post-Training Quantization (PTQ)."""

    enabled: bool = True
    backend: str = "fbgemm"
    calibration_samples: int = 512
    calibration_batch_size: int = 64


@dataclass
class PruningConfig:
    """Configuration for magnitude pruning and post-pruning fine-tuning."""

    enabled: bool = True
    sparsity_levels: list[float] = field(default_factory=lambda: [0.10, 0.20, 0.30])
    method: str = "l1_unstructured"
    fine_tune_epochs: int = 3
    learning_rate: float = 0.0001


@dataclass
class DistillationConfig:
    """Configuration for Knowledge Distillation (Teacher -> Student)."""

    enabled: bool = True
    student_model_name: str = "mobilenet_v3_small"
    temperature: float = 4.0
    alpha: float = 0.5
    epochs: int = 5
    learning_rate: float = 0.0003
    batch_size: int = 64


@dataclass
class QualityGateConfig:
    """Acceptance criteria and thresholds for selecting an Optimized Champion."""

    max_macro_f1_drop: float = 0.03
    max_accuracy_drop: float = 0.03
    min_latency_reduction_pct: float = 5.0
    min_size_reduction_pct: float = 10.0


@dataclass
class ExportConfig:
    """Configuration for model deployment export (e.g. ONNX)."""

    enabled: bool = True
    format: str = "onnx"
    opset_version: int = 17
    dynamic_axes: bool = True
    atol: float = 1e-3


@dataclass
class OptimizationPipelineConfig:
    """Master configuration for Phase 08 optimization experiments."""

    experiment_name: str = "champion_optimization"
    seed: int = 42
    reference_model: str = "resnet18"
    reference_checkpoint: str = (
        "artifacts/training/resnet18/resnet18_training_20260817_203647/best.pt"
    )
    data_path: str = "data/raw/fer2013/fer2013.csv"
    output_dir: str = "artifacts/optimized"
    fp16: FP16Config = field(default_factory=FP16Config)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationPipelineConfig:
        """Construct configuration from raw dictionary."""
        fp16_data = data.get("fp16", {})
        quant_data = data.get("quantization", {})
        prune_data = data.get("pruning", {})
        dist_data = data.get("distillation", {})
        gate_data = data.get("quality_gate", {})
        export_data = data.get("export", {})

        return cls(
            experiment_name=str(data.get("experiment_name", "champion_optimization")),
            seed=int(data.get("seed", 42)),
            reference_model=str(data.get("reference_model", "resnet18")),
            reference_checkpoint=str(data.get("reference_checkpoint", "")),
            data_path=str(data.get("data_path", "data/raw/fer2013/fer2013.csv")),
            output_dir=str(data.get("output_dir", "artifacts/optimized")),
            fp16=FP16Config(**fp16_data),
            quantization=QuantizationConfig(**quant_data),
            pruning=PruningConfig(**prune_data),
            distillation=DistillationConfig(**dist_data),
            quality_gate=QualityGateConfig(**gate_data),
            export=ExportConfig(**export_data),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> OptimizationPipelineConfig:
        """Load configuration from a YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Optimization configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Invalid YAML content in {config_path}: expected dictionary")

        return cls.from_dict(raw)


def load_optimization_config(path: str | Path | None = None) -> OptimizationPipelineConfig:
    """Load optimization config from specified path or default location."""
    if path is not None:
        return OptimizationPipelineConfig.from_yaml(path)

    default_path = Path("ml/configs/optimization.yaml")
    if default_path.exists():
        return OptimizationPipelineConfig.from_yaml(default_path)

    return OptimizationPipelineConfig()
