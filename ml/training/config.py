"""Training configuration schema and validation dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer."""

    name: str = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0.0, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0.0, got {self.weight_decay}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0.0, got {self.eps}")
        if (
            len(self.betas) != 2
            or not (0.0 <= self.betas[0] < 1.0)
            or not (0.0 <= self.betas[1] < 1.0)
        ):
            raise ValueError(f"betas must be two floats in [0, 1), got {self.betas}")


@dataclass(frozen=True)
class LossConfig:
    """Configuration for loss function."""

    name: str = "cross_entropy"
    class_weighted: bool = False


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    name: str = "reduce_on_plateau"
    mode: Literal["min", "max"] = "min"
    factor: float = 0.5
    patience: int = 2
    min_lr: float = 1e-6
    step_size: int = 10
    gamma: float = 0.1
    t_max: int = 30

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError(f"scheduler mode must be 'min' or 'max', got {self.mode}")
        if not (0.0 < self.factor < 1.0):
            raise ValueError(f"scheduler factor must be in (0, 1), got {self.factor}")
        if self.patience < 0:
            raise ValueError(f"scheduler patience must be >= 0, got {self.patience}")
        if self.min_lr < 0.0:
            raise ValueError(f"scheduler min_lr must be >= 0.0, got {self.min_lr}")


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Configuration for early stopping."""

    enabled: bool = True
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    patience: int = 5
    min_delta: float = 0.001

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError(f"early stopping mode must be 'min' or 'max', got {self.mode}")
        if self.patience < 0:
            raise ValueError(f"early stopping patience must be >= 0, got {self.patience}")
        if self.min_delta < 0.0:
            raise ValueError(f"early stopping min_delta must be >= 0.0, got {self.min_delta}")


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for model checkpointing."""

    save_best: bool = True
    save_last: bool = True
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    save_dir: str = "artifacts/training/baseline_cnn"

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError(f"checkpoint mode must be 'min' or 'max', got {self.mode}")


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training loop."""

    epochs: int = 30
    batch_size: int = 64
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    mixed_precision: bool = False
    gradient_clipping: bool = False
    max_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.gradient_clipping and self.max_norm <= 0.0:
            raise ValueError(
                f"max_norm must be > 0.0 when gradient clipping is enabled, got {self.max_norm}"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    """Metadata describing the experiment run."""

    name: str = "baseline_cnn"
    version: str = "v1"
    description: str = "Baseline CNN training pipeline"


@dataclass(frozen=True)
class DataConfig:
    """Data loading configuration."""

    raw_path: str = "data/raw/fer2013/fer2013.csv"
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"data batch_size must be > 0, got {self.batch_size}")


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Reproducibility configuration."""

    seed: int = 42
    deterministic: bool = True


@dataclass(frozen=True)
class TrainingPipelineConfig:
    """Full top-level pipeline configuration."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model_name: str = "baseline_cnn"
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingPipelineConfig:
        """Create TrainingPipelineConfig from a dictionary."""
        exp_dict = data.get("experiment", {})
        exp_config = ExperimentConfig(
            name=exp_dict.get("name", "baseline_cnn"),
            version=exp_dict.get("version", "v1"),
            description=exp_dict.get("description", "Baseline CNN training pipeline"),
        )

        model_dict = data.get("model", {})
        model_name = (
            model_dict.get("name", "baseline_cnn")
            if isinstance(model_dict, dict)
            else str(model_dict)
        )

        data_dict = data.get("data", {})
        data_config = DataConfig(
            raw_path=data_dict.get("raw_path", "data/raw/fer2013/fer2013.csv"),
            batch_size=int(data_dict.get("batch_size", 64)),
            num_workers=int(data_dict.get("num_workers", 0)),
            pin_memory=bool(data_dict.get("pin_memory", False)),
        )

        tr_dict = data.get("training", {})

        opt_dict = tr_dict.get("optimizer", {})
        opt_betas = opt_dict.get("betas", [0.9, 0.999])
        optimizer_config = OptimizerConfig(
            name=opt_dict.get("name", "adamw"),
            learning_rate=float(opt_dict.get("learning_rate", 0.001)),
            weight_decay=float(opt_dict.get("weight_decay", 0.0001)),
            betas=(float(opt_betas[0]), float(opt_betas[1])),
            eps=float(opt_dict.get("eps", 1e-8)),
        )

        loss_dict = tr_dict.get("loss", {})
        loss_config = LossConfig(
            name=loss_dict.get("name", "cross_entropy"),
            class_weighted=bool(loss_dict.get("class_weighted", False)),
        )

        sched_dict = tr_dict.get("scheduler", {})
        scheduler_config = SchedulerConfig(
            name=sched_dict.get("name", "reduce_on_plateau"),
            mode=sched_dict.get("mode", "min"),
            factor=float(sched_dict.get("factor", 0.5)),
            patience=int(sched_dict.get("patience", 2)),
            min_lr=float(sched_dict.get("min_lr", 1e-6)),
            step_size=int(sched_dict.get("step_size", 10)),
            gamma=float(sched_dict.get("gamma", 0.1)),
            t_max=int(sched_dict.get("t_max", 30)),
        )

        es_dict = tr_dict.get("early_stopping", {})
        early_stopping_config = EarlyStoppingConfig(
            enabled=bool(es_dict.get("enabled", True)),
            monitor=es_dict.get("monitor", "val_loss"),
            mode=es_dict.get("mode", "min"),
            patience=int(es_dict.get("patience", 5)),
            min_delta=float(es_dict.get("min_delta", 0.001)),
        )

        ckpt_dict = tr_dict.get("checkpoint", {})
        checkpoint_config = CheckpointConfig(
            save_best=bool(ckpt_dict.get("save_best", True)),
            save_last=bool(ckpt_dict.get("save_last", True)),
            monitor=ckpt_dict.get("monitor", "val_loss"),
            mode=ckpt_dict.get("mode", "min"),
            save_dir=ckpt_dict.get("save_dir", "artifacts/training/baseline_cnn"),
        )

        mp_dict = tr_dict.get("mixed_precision", False)
        mp_enabled = mp_dict.get("enabled", False) if isinstance(mp_dict, dict) else bool(mp_dict)

        gc_dict = tr_dict.get("gradient_clipping", False)
        if isinstance(gc_dict, dict):
            gc_enabled = bool(gc_dict.get("enabled", False))
            max_norm = float(gc_dict.get("max_norm", 1.0))
        else:
            gc_enabled = bool(gc_dict)
            max_norm = 1.0

        training_config = TrainingConfig(
            epochs=int(tr_dict.get("epochs", 30)),
            batch_size=int(tr_dict.get("batch_size", 64)),
            optimizer=optimizer_config,
            loss=loss_config,
            scheduler=scheduler_config,
            early_stopping=early_stopping_config,
            checkpoint=checkpoint_config,
            mixed_precision=mp_enabled,
            gradient_clipping=gc_enabled,
            max_norm=max_norm,
        )

        rep_dict = data.get("reproducibility", {})
        reproducibility_config = ReproducibilityConfig(
            seed=int(rep_dict.get("seed", 42)),
            deterministic=bool(rep_dict.get("deterministic", True)),
        )

        return cls(
            experiment=exp_config,
            model_name=model_name,
            data=data_config,
            training=training_config,
            reproducibility=reproducibility_config,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingPipelineConfig:
        """Load and parse configuration from a YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Training configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        if not isinstance(raw_data, dict):
            raise ValueError(f"Invalid YAML content in {config_path}: expected dictionary")

        return cls.from_dict(raw_data)


def load_training_config(path: str | Path | None = None) -> TrainingPipelineConfig:
    """Load training configuration from given path or default path."""
    if path is not None:
        return TrainingPipelineConfig.from_yaml(path)

    default_path = Path("ml/configs/training/baseline.yaml")
    if not default_path.exists():
        default_path = Path("ml/configs/training.yaml")

    return TrainingPipelineConfig.from_yaml(default_path)
