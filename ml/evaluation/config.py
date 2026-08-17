"""Dataclass schema and parser for evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for evaluation experiment metadata."""

    name: str = "baseline_cnn_eval"
    version: str = "v1"
    description: str = "Baseline CNN test evaluation"


@dataclass(frozen=True)
class CheckpointEvaluationConfig:
    """Configuration for checkpoint to evaluate."""

    path: str = "artifacts/training/baseline_cnn/smoke_test_20260817_055905/best.pt"
    type: Literal["best", "last", "custom"] = "best"

    def __post_init__(self) -> None:
        if not self.path or not self.path.strip():
            raise ValueError("Checkpoint path cannot be empty")


@dataclass(frozen=True)
class EvaluationDataConfig:
    """Configuration for evaluation data split and DataLoader."""

    raw_path: str = "data/raw/fer2013/fer2013.csv"
    split: Literal["test", "val", "train"] = "test"
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.split not in ("test", "val", "train"):
            raise ValueError(f"split must be 'test', 'val', or 'train', got '{self.split}'")


@dataclass(frozen=True)
class MetricsEvaluationConfig:
    """Configuration for metric flags."""

    accuracy: bool = True
    precision: bool = True
    recall: bool = True
    f1: bool = True
    confusion_matrix: bool = True
    classification_report: bool = True


@dataclass(frozen=True)
class ConfidenceConfig:
    """Configuration for confidence analysis."""

    enabled: bool = True
    high_confidence_threshold: float = 0.90
    num_bins: int = 10

    def __post_init__(self) -> None:
        if not (0.0 < self.high_confidence_threshold <= 1.0):
            raise ValueError(
                f"high_confidence_threshold must be in (0, 1], got {self.high_confidence_threshold}"
            )
        if self.num_bins <= 0:
            raise ValueError(f"num_bins must be > 0, got {self.num_bins}")


@dataclass(frozen=True)
class ErrorAnalysisConfig:
    """Configuration for error analysis and recording."""

    enabled: bool = True
    save_incorrect: bool = True
    max_samples_recorded: int | None = None

    def __post_init__(self) -> None:
        if self.max_samples_recorded is not None and self.max_samples_recorded <= 0:
            raise ValueError(
                f"max_samples_recorded must be > 0 if specified, got {self.max_samples_recorded}"
            )


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for inference latency and throughput benchmarking."""

    enabled: bool = True
    warmup_iterations: int = 10
    benchmark_iterations: int = 50
    batch_size: int = 64
    single_sample: bool = True

    def __post_init__(self) -> None:
        if self.warmup_iterations < 0:
            raise ValueError(f"warmup_iterations must be >= 0, got {self.warmup_iterations}")
        if self.benchmark_iterations <= 0:
            raise ValueError(f"benchmark_iterations must be > 0, got {self.benchmark_iterations}")
        if self.batch_size <= 0:
            raise ValueError(f"benchmark batch_size must be > 0, got {self.batch_size}")


@dataclass(frozen=True)
class ArtifactsConfig:
    """Configuration for evaluation artifact storage."""

    save_dir: str = "artifacts/evaluation/baseline_cnn"
    save_visualizations: bool = True


@dataclass(frozen=True)
class EvaluationPipelineConfig:
    """Root configuration for model evaluation pipeline."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model_name: str = "baseline_cnn"
    checkpoint: CheckpointEvaluationConfig = field(default_factory=CheckpointEvaluationConfig)
    data: EvaluationDataConfig = field(default_factory=EvaluationDataConfig)
    metrics: MetricsEvaluationConfig = field(default_factory=MetricsEvaluationConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    error_analysis: ErrorAnalysisConfig = field(default_factory=ErrorAnalysisConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)

    def __post_init__(self) -> None:
        if not self.model_name or not self.model_name.strip():
            raise ValueError("model_name cannot be empty")


def load_evaluation_config(config_path: str | Path) -> EvaluationPipelineConfig:
    """Load and parse evaluation configuration from YAML file.

    Args:
        config_path: Path to evaluation YAML configuration file.

    Returns:
        Validated EvaluationPipelineConfig instance.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation configuration file not found at: {path}")

    with open(path, encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    exp_dict = raw_dict.get("experiment", {})
    ckpt_dict = raw_dict.get("checkpoint", {})
    data_dict = raw_dict.get("data", {})
    metrics_dict = raw_dict.get("metrics", {})
    conf_dict = raw_dict.get("confidence", {})
    err_dict = raw_dict.get("error_analysis", {})
    bench_dict = raw_dict.get("benchmark", {})
    art_dict = raw_dict.get("artifacts", {})

    # Map literal split and checkpoint type
    raw_split = str(data_dict.get("split", "test")).lower().strip()
    split_val: Literal["test", "val", "train"] = (
        "val" if raw_split == "val" else ("train" if raw_split == "train" else "test")
    )

    raw_ckpt_type = str(ckpt_dict.get("type", "best")).lower().strip()
    ckpt_type_val: Literal["best", "last", "custom"] = (
        "last" if raw_ckpt_type == "last" else ("custom" if raw_ckpt_type == "custom" else "best")
    )

    return EvaluationPipelineConfig(
        experiment=ExperimentConfig(
            name=str(exp_dict.get("name", "baseline_cnn_eval")),
            version=str(exp_dict.get("version", "v1")),
            description=str(exp_dict.get("description", "Baseline CNN test evaluation")),
        ),
        model_name=str(
            raw_dict.get("model", {}).get("name", raw_dict.get("model_name", "baseline_cnn"))
        ),
        checkpoint=CheckpointEvaluationConfig(
            path=str(
                ckpt_dict.get(
                    "path",
                    "artifacts/training/baseline_cnn/smoke_test_20260817_055905/best.pt",
                )
            ),
            type=ckpt_type_val,
        ),
        data=EvaluationDataConfig(
            raw_path=str(data_dict.get("raw_path", "data/raw/fer2013/fer2013.csv")),
            split=split_val,
            batch_size=int(data_dict.get("batch_size", 64)),
            num_workers=int(data_dict.get("num_workers", 0)),
            pin_memory=bool(data_dict.get("pin_memory", False)),
        ),
        metrics=MetricsEvaluationConfig(
            accuracy=bool(metrics_dict.get("accuracy", True)),
            precision=bool(metrics_dict.get("precision", True)),
            recall=bool(metrics_dict.get("recall", True)),
            f1=bool(metrics_dict.get("f1", True)),
            confusion_matrix=bool(metrics_dict.get("confusion_matrix", True)),
            classification_report=bool(metrics_dict.get("classification_report", True)),
        ),
        confidence=ConfidenceConfig(
            enabled=bool(conf_dict.get("enabled", True)),
            high_confidence_threshold=float(conf_dict.get("high_confidence_threshold", 0.90)),
            num_bins=int(conf_dict.get("num_bins", 10)),
        ),
        error_analysis=ErrorAnalysisConfig(
            enabled=bool(err_dict.get("enabled", True)),
            save_incorrect=bool(err_dict.get("save_incorrect", True)),
            max_samples_recorded=err_dict.get("max_samples_recorded"),
        ),
        benchmark=BenchmarkConfig(
            enabled=bool(bench_dict.get("enabled", True)),
            warmup_iterations=int(bench_dict.get("warmup_iterations", 10)),
            benchmark_iterations=int(bench_dict.get("benchmark_iterations", 50)),
            batch_size=int(bench_dict.get("batch_size", 64)),
            single_sample=bool(bench_dict.get("single_sample", True)),
        ),
        artifacts=ArtifactsConfig(
            save_dir=str(art_dict.get("save_dir", "artifacts/evaluation/baseline_cnn")),
            save_visualizations=bool(art_dict.get("save_visualizations", True)),
        ),
    )
