"""Atomic model checkpoint management, saving, loading, and state restoration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from ml.training.config import CheckpointConfig

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving, loading, and atomic file operations for model checkpoints."""

    def __init__(
        self,
        save_dir: str | Path = "artifacts/training/baseline_cnn",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best: bool = True,
        save_last: bool = True,
    ) -> None:
        """Initialize CheckpointManager.

        Args:
            save_dir: Directory where checkpoints will be stored.
            monitor: Name of metric being monitored (e.g. 'val_loss').
            mode: 'min' or 'max' for the monitored metric.
            save_best: Whether to maintain a best.pt checkpoint.
            save_last: Whether to maintain a last.pt checkpoint.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last

        self.best_metric: float | None = None
        self.best_epoch: int = 0

    @classmethod
    def from_config(
        cls, config: CheckpointConfig, run_dir: Path | None = None
    ) -> CheckpointManager:
        """Instantiate CheckpointManager from CheckpointConfig."""
        target_dir = run_dir if run_dir is not None else Path(config.save_dir)
        return cls(
            save_dir=target_dir,
            monitor=config.monitor,
            mode=config.mode,
            save_best=config.save_best,
            save_last=config.save_last,
        )

    def is_better(self, metric: float) -> bool:
        """Determine whether the given metric improves upon current best."""
        if self.best_metric is None:
            return True
        if self.mode == "min":
            return metric < self.best_metric
        return metric > self.best_metric

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer | None,
        scheduler: Any | None,
        epoch: int,
        metric: float,
        history: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
        scaler: Any | None = None,
        is_best_override: bool | None = None,
    ) -> dict[str, Path]:
        """Save checkpoint state dictionary atomically.

        Args:
            model: PyTorch model.
            optimizer: Optimizer instance or None.
            scheduler: Learning rate scheduler or None.
            epoch: Current epoch integer.
            metric: Monitored metric value.
            history: List of completed epoch metric dictionaries.
            config: Serialized configuration dictionary.
            scaler: Optional GradScaler for mixed precision.
            is_best_override: Explicit boolean to force best checkpoint save.

        Returns:
            Dictionary of saved checkpoint file paths.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_data: dict[str, Any] = {
            "epoch": epoch,
            "metric": metric,
            "monitor": self.monitor,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "history": history,
            "config": config or {},
        }

        saved_paths: dict[str, Path] = {}

        # Determine if this is the best model
        is_best = self.is_better(metric) if is_best_override is None else is_best_override
        if is_best:
            self.best_metric = metric
            self.best_epoch = epoch
            checkpoint_data["best_metric"] = self.best_metric
            checkpoint_data["best_epoch"] = self.best_epoch

        # Save last checkpoint
        if self.save_last:
            last_path = self.save_dir / "last.pt"
            self._atomic_save(checkpoint_data, last_path)
            saved_paths["last"] = last_path

        # Save best checkpoint
        if self.save_best and is_best:
            best_path = self.save_dir / "best.pt"
            self._atomic_save(checkpoint_data, best_path)
            saved_paths["best"] = best_path
            logger.info(
                f"Saved new best checkpoint to {best_path} (epoch {epoch}, {self.monitor}={metric:.6f})"
            )

        return saved_paths

    def _atomic_save(self, state: dict[str, Any], target_path: Path) -> None:
        """Write checkpoint to a temporary file first then atomically replace target."""
        temp_path = target_path.with_suffix(".tmp")
        torch.save(state, temp_path)
        if temp_path.exists():
            if target_path.exists():
                os.replace(temp_path, target_path)
            else:
                temp_path.rename(target_path)

    @staticmethod
    def load(
        checkpoint_path: str | Path,
        model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Load checkpoint file and restore weights/states into provided objects.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            model: Optional model instance to load model_state_dict into.
            optimizer: Optional optimizer instance to load optimizer_state_dict into.
            scheduler: Optional scheduler instance to load scheduler_state_dict into.
            scaler: Optional GradScaler instance to load scaler_state_dict into.
            device: Compute device to map tensors to.

        Returns:
            The loaded checkpoint dictionary.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = (
            torch.save if False else torch.load(path, map_location=device, weights_only=False)
        )

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint format in {path}: expected dict")

        if model is not None and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return checkpoint
