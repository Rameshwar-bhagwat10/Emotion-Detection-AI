"""Core Trainer class for model training, validation, and lifecycle orchestration."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ml.training.checkpointing import CheckpointManager
from ml.training.config import TrainingPipelineConfig
from ml.training.early_stopping import EarlyStopping
from ml.training.metrics import MetricTracker

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates model training, validation, checkpointing, and lifecycle management."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler: Any | None = None,
        is_scheduler_metric_dependent: bool = True,
        device: torch.device | str = "cpu",
        config: TrainingPipelineConfig | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        early_stopping: EarlyStopping | None = None,
        run_dir: Path | str | None = None,
        use_amp: bool = False,
        gradient_clipping: bool = False,
        max_norm: float = 1.0,
        custom_logger: logging.Logger | None = None,
    ) -> None:
        """Initialize Trainer with injected components.

        Args:
            model: PyTorch neural network model.
            optimizer: Configured optimizer instance.
            criterion: Configured loss function.
            train_loader: DataLoader for training dataset split.
            val_loader: DataLoader for validation dataset split.
            scheduler: Optional learning rate scheduler.
            is_scheduler_metric_dependent: True if scheduler.step() requires val_metric.
            device: Compute device ('cpu', 'cuda', etc.).
            config: Full training pipeline configuration.
            checkpoint_manager: Optional CheckpointManager instance.
            early_stopping: Optional EarlyStopping instance.
            run_dir: Directory for storing run artifacts and checkpoints.
            use_amp: Whether to use automatic mixed precision (CUDA only).
            gradient_clipping: Whether to clip gradient norms.
            max_norm: Maximum gradient norm for clipping.
            custom_logger: Optional logger instance.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.is_scheduler_metric_dependent = is_scheduler_metric_dependent
        self.config = config
        self.logger = custom_logger or logger

        self.run_dir = (
            Path(run_dir) if run_dir is not None else Path("artifacts/training/baseline_cnn")
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_manager = checkpoint_manager or CheckpointManager(save_dir=self.run_dir)
        self.early_stopping = early_stopping

        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.gradient_clipping = gradient_clipping
        self.max_norm = max_norm

        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
        self.history: list[dict[str, Any]] = []

    def get_current_lr(self) -> float:
        """Retrieve current learning rate from optimizer."""
        return float(self.optimizer.param_groups[0]["lr"])

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Execute one complete training epoch over train_loader.

        Args:
            epoch: Current epoch index.

        Returns:
            Dictionary with 'loss', 'accuracy', and 'learning_rate'.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                if self.gradient_clipping:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()

            # Record batch metrics using scalar loss
            self.train_metrics.update(
                loss_val=loss.item(),
                logits=logits.detach(),
                labels=labels.detach(),
                batch_size=images.shape[0],
            )

        computed = self.train_metrics.compute()
        computed["learning_rate"] = self.get_current_lr()
        return computed

    def validate_epoch(self, epoch: int) -> dict[str, float]:
        """Execute one complete validation epoch over val_loader without gradient computation.

        Args:
            epoch: Current epoch index.

        Returns:
            Dictionary with 'loss' and 'accuracy'.
        """
        self.model.eval()
        self.val_metrics.reset()

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                self.val_metrics.update(
                    loss_val=loss.item(),
                    logits=logits,
                    labels=labels,
                    batch_size=images.shape[0],
                )

        return self.val_metrics.compute()

    def fit(self, epochs: int, start_epoch: int = 1) -> dict[str, Any]:
        """Execute complete training and validation cycle across configured epochs.

        Args:
            epochs: Total number of epochs to train.
            start_epoch: Epoch number to start on (defaults to 1, or restored value when resuming).

        Returns:
            Summary dictionary containing history and best metric info.
        """
        self.logger.info(
            f"Starting training run: epochs={epochs}, start_epoch={start_epoch}, "
            f"device={self.device}, run_dir={self.run_dir}"
        )

        # Save configuration snapshot at training start
        if self.config is not None:
            self._save_config_snapshot()

        best_epoch = (
            self.checkpoint_manager.best_epoch
            if self.checkpoint_manager.best_epoch > 0
            else start_epoch
        )
        best_val_loss = (
            self.checkpoint_manager.best_metric
            if self.checkpoint_manager.best_metric is not None
            else float("inf")
        )
        best_val_acc = 0.0

        for epoch in range(start_epoch, epochs + 1):
            epoch_start_time = time.time()

            current_lr = self.get_current_lr()
            train_res = self.train_epoch(epoch)
            val_res = self.validate_epoch(epoch)

            # Step scheduler
            if self.scheduler is not None:
                if self.is_scheduler_metric_dependent:
                    self.scheduler.step(val_res["loss"])
                else:
                    self.scheduler.step()

            duration = round(time.time() - epoch_start_time, 2)

            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": round(train_res["loss"], 6),
                "train_accuracy": round(train_res["accuracy"], 6),
                "val_loss": round(val_res["loss"], 6),
                "val_accuracy": round(val_res["accuracy"], 6),
                "learning_rate": current_lr,
                "epoch_duration_seconds": duration,
            }
            self.history.append(epoch_record)

            self.logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_res['loss']:.4f} | Train Acc: {train_res['accuracy']*100:.2f}% | "
                f"Val Loss: {val_res['loss']:.4f} | Val Acc: {val_res['accuracy']*100:.2f}% | "
                f"LR: {current_lr:.6f} | Time: {duration}s"
            )

            # Save checkpoints
            config_dict = self._get_config_dict()
            saved_paths = self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metric=val_res["loss"],
                history=self.history,
                config=config_dict,
                scaler=self.scaler,
            )

            if "best" in saved_paths:
                best_epoch = epoch
                best_val_loss = val_res["loss"]
                best_val_acc = val_res["accuracy"]

            # Persist training history
            self._save_history()

            # Early stopping check
            if self.early_stopping is not None:
                _, should_stop = self.early_stopping.step(val_res["loss"], epoch=epoch)
                if should_stop:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch}. Terminating training run."
                    )
                    break

        self.logger.info(
            f"Training run complete. Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | "
            f"Best Val Acc: {best_val_acc*100:.2f}%"
        )

        return {
            "history": self.history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_acc,
            "final_train_loss": self.history[-1]["train_loss"] if self.history else 0.0,
            "final_train_accuracy": self.history[-1]["train_accuracy"] if self.history else 0.0,
            "final_val_loss": self.history[-1]["val_loss"] if self.history else 0.0,
            "final_val_accuracy": self.history[-1]["val_accuracy"] if self.history else 0.0,
        }

    def _save_history(self) -> None:
        """Persist structured training history to history.json."""
        history_path = self.run_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def _save_config_snapshot(self) -> None:
        """Persist snapshot of training configuration to config.yaml."""
        config_path = self.run_dir / "config.yaml"
        config_dict = self._get_config_dict()
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def _get_config_dict(self) -> dict[str, Any]:
        """Convert config to dictionary representation."""
        if self.config is None:
            return {}
        return {
            "experiment": {
                "name": self.config.experiment.name,
                "version": self.config.experiment.version,
                "description": self.config.experiment.description,
            },
            "model": {"name": self.config.model_name},
            "data": {
                "raw_path": self.config.data.raw_path,
                "batch_size": self.config.data.batch_size,
                "num_workers": self.config.data.num_workers,
                "pin_memory": self.config.data.pin_memory,
            },
            "training": {
                "epochs": self.config.training.epochs,
                "batch_size": self.config.training.batch_size,
                "optimizer": {
                    "name": self.config.training.optimizer.name,
                    "learning_rate": self.config.training.optimizer.learning_rate,
                    "weight_decay": self.config.training.optimizer.weight_decay,
                    "betas": list(self.config.training.optimizer.betas),
                    "eps": self.config.training.optimizer.eps,
                },
                "loss": {
                    "name": self.config.training.loss.name,
                    "class_weighted": self.config.training.loss.class_weighted,
                },
                "scheduler": {
                    "name": self.config.training.scheduler.name,
                    "mode": self.config.training.scheduler.mode,
                    "factor": self.config.training.scheduler.factor,
                    "patience": self.config.training.scheduler.patience,
                    "min_lr": self.config.training.scheduler.min_lr,
                },
                "early_stopping": {
                    "enabled": self.config.training.early_stopping.enabled,
                    "monitor": self.config.training.early_stopping.monitor,
                    "mode": self.config.training.early_stopping.mode,
                    "patience": self.config.training.early_stopping.patience,
                    "min_delta": self.config.training.early_stopping.min_delta,
                },
                "checkpoint": {
                    "save_best": self.config.training.checkpoint.save_best,
                    "save_last": self.config.training.checkpoint.save_last,
                    "monitor": self.config.training.checkpoint.monitor,
                    "mode": self.config.training.checkpoint.mode,
                    "save_dir": self.config.training.checkpoint.save_dir,
                },
                "mixed_precision": self.config.training.mixed_precision,
                "gradient_clipping": self.config.training.gradient_clipping,
                "max_norm": self.config.training.max_norm,
            },
            "reproducibility": {
                "seed": self.config.reproducibility.seed,
                "deterministic": self.config.reproducibility.deterministic,
            },
        }
