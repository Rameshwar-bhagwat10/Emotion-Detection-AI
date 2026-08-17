"""Learning rate scheduler factory and wrapper utilities."""

from __future__ import annotations

from typing import Any, Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)

from ml.training.config import SchedulerConfig


def create_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig | dict[str, Any],
) -> tuple[Any, bool]:
    """Create and return a configured PyTorch learning rate scheduler.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        config: SchedulerConfig instance or configuration dictionary.

    Returns:
        Tuple of (scheduler_instance, is_metric_dependent_bool).
        When is_metric_dependent is True, scheduler.step(val_metric) must be called.
    """
    if isinstance(config, dict):
        raw_mode = str(config.get("mode", "min")).lower().strip()
        mode_val: Literal["min", "max"] = "max" if raw_mode == "max" else "min"
        sched_config = SchedulerConfig(
            name=str(config.get("name", "reduce_on_plateau")).lower().strip(),
            mode=mode_val,
            factor=float(config.get("factor", 0.5)),
            patience=int(config.get("patience", 2)),
            min_lr=float(config.get("min_lr", 1e-6)),
            step_size=int(config.get("step_size", 10)),
            gamma=float(config.get("gamma", 0.1)),
            t_max=int(config.get("t_max", 30)),
        )
    elif isinstance(config, SchedulerConfig):
        sched_config = config
    else:
        raise TypeError(f"Unsupported config type for create_scheduler: {type(config)}")

    name = sched_config.name.lower().strip()

    if name in ("reduce_on_plateau", "reducelronplateau", "plateau"):
        plateau_scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode=sched_config.mode,
            factor=sched_config.factor,
            patience=sched_config.patience,
            min_lr=sched_config.min_lr,
        )
        return plateau_scheduler, True

    elif name in ("step_lr", "steplr", "step"):
        step_scheduler = StepLR(
            optimizer=optimizer,
            step_size=sched_config.step_size,
            gamma=sched_config.gamma,
        )
        return step_scheduler, False

    elif name in ("cosine_annealing", "cosine", "cosineannealinglr"):
        cosine_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=sched_config.t_max,
            eta_min=sched_config.min_lr,
        )
        return cosine_scheduler, False

    raise ValueError(
        f"Unsupported scheduler '{name}'. Supported schedulers: ['reduce_on_plateau', 'step_lr', 'cosine_annealing']"
    )
