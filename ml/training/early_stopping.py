"""Early stopping tracker for model training."""

from __future__ import annotations

import logging

from ml.training.config import EarlyStoppingConfig

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Monitors validation metrics to stop training when improvement plateaus."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
        enabled: bool = True,
    ) -> None:
        """Initialize early stopping controller.

        Args:
            patience: Number of epochs to wait for improvement before stopping.
            min_delta: Minimum change in monitored metric to qualify as an improvement.
            mode: 'min' if lower is better (e.g. loss), 'max' if higher is better (e.g. accuracy).
            enabled: If False, early stopping is bypassed.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")
        if min_delta < 0.0:
            raise ValueError(f"min_delta must be >= 0.0, got {min_delta}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.enabled = enabled

        self.best_score: float | None = None
        self.patience_counter: int = 0
        self.best_epoch: int = 0
        self.is_triggered: bool = False

    @classmethod
    def from_config(cls, config: EarlyStoppingConfig) -> EarlyStopping:
        """Instantiate EarlyStopping from an EarlyStoppingConfig object."""
        return cls(
            patience=config.patience,
            min_delta=config.min_delta,
            mode=config.mode,
            enabled=config.enabled,
        )

    def step(self, metric: float, epoch: int = 0) -> tuple[bool, bool]:
        """Update tracker with the latest epoch metric.

        Args:
            metric: The current value of the monitored metric.
            epoch: The current epoch index.

        Returns:
            Tuple of (is_improvement, should_stop).
        """
        if not self.enabled:
            return True, False

        if self.best_score is None:
            self.best_score = metric
            self.best_epoch = epoch
            self.patience_counter = 0
            return True, False

        if self.mode == "min":
            improvement = (self.best_score - metric) >= self.min_delta
        else:
            improvement = (metric - self.best_score) >= self.min_delta

        if improvement:
            self.best_score = metric
            self.best_epoch = epoch
            self.patience_counter = 0
            return True, False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.is_triggered = True
                logger.info(
                    f"Early stopping triggered at epoch {epoch}: "
                    f"no improvement over best {self.mode} metric ({self.best_score:.6f} at epoch {self.best_epoch}) "
                    f"for {self.patience_counter} consecutive epochs."
                )
                return False, True
            return False, False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_score = None
        self.patience_counter = 0
        self.best_epoch = 0
        self.is_triggered = False
