"""Integration tests for Phase 05 Baseline CNN Training Pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.models.factory import create_model
from ml.preprocessing.dataloaders import build_dataloaders
from ml.training.checkpointing import CheckpointManager
from ml.training.config import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    TrainingPipelineConfig,
)
from ml.training.early_stopping import EarlyStopping
from ml.training.losses import create_loss
from ml.training.optimizers import create_optimizer
from ml.training.schedulers import create_scheduler
from ml.training.trainer import Trainer

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def create_synthetic_dataloaders(batch_size: int = 4) -> tuple[DataLoader, DataLoader]:
    """Create lightweight synthetic DataLoaders for fast deterministic integration tests."""
    train_x = torch.randn(16, 1, 48, 48, dtype=torch.float32)
    train_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1], dtype=torch.long)
    train_ds = TensorDataset(train_x, train_y)

    val_x = torch.randn(8, 1, 48, 48, dtype=torch.float32)
    val_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0], dtype=torch.long)
    val_ds = TensorDataset(val_x, val_y)

    def collate_fn(batch):
        xs = torch.stack([item[0] for item in batch])
        ys = torch.stack([item[1] for item in batch])
        return {"image": xs, "label": ys}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


def test_training_step_parameter_mutation() -> None:
    """Verify that a single training epoch actually mutates trainable model parameters."""
    model = create_model("baseline_cnn")
    initial_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    train_loader, val_loader = create_synthetic_dataloaders(batch_size=4)
    optimizer = create_optimizer(
        model.parameters(), OptimizerConfig(name="adamw", learning_rate=0.01)
    )
    criterion = create_loss("cross_entropy")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
    )

    metrics = trainer.train_epoch(epoch=1)
    assert "loss" in metrics and "accuracy" in metrics
    assert metrics["loss"] >= 0.0

    updated_params = [p for p in model.parameters() if p.requires_grad]
    has_changed = any(
        not torch.equal(p_init, p_up)
        for p_init, p_up in zip(initial_params, updated_params, strict=True)
    )
    assert has_changed, "Training step failed to update model parameters"


def test_validation_weight_immutability() -> None:
    """Verify that validation step does NOT mutate model parameters."""
    model = create_model("baseline_cnn")
    initial_params = [p.clone().detach() for p in model.parameters()]

    train_loader, val_loader = create_synthetic_dataloaders(batch_size=4)
    optimizer = create_optimizer(
        model.parameters(), OptimizerConfig(name="adamw", learning_rate=0.01)
    )
    criterion = create_loss("cross_entropy")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
    )

    metrics = trainer.validate_epoch(epoch=1)
    assert "loss" in metrics and "accuracy" in metrics

    for p_init, p_curr in zip(initial_params, model.parameters(), strict=True):
        assert torch.equal(p_init, p_curr), "Validation modified model parameters!"


def test_end_to_end_fit_and_resume(tmp_path: Path) -> None:
    """Verify full fit cycle, checkpoint creation, and resuming to completion."""
    model = create_model("baseline_cnn")
    train_loader, val_loader = create_synthetic_dataloaders(batch_size=4)
    optimizer = create_optimizer(
        model.parameters(), OptimizerConfig(name="adamw", learning_rate=0.001)
    )
    criterion = create_loss("cross_entropy")
    scheduler, is_metric_dep = create_scheduler(
        optimizer, SchedulerConfig(name="reduce_on_plateau", patience=1)
    )

    ckpt_manager = CheckpointManager(save_dir=tmp_path)
    early_stopping = EarlyStopping(patience=3)

    pipeline_config = TrainingPipelineConfig(
        model_name="baseline_cnn",
        training=TrainingConfig(epochs=2, batch_size=4),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        is_scheduler_metric_dependent=is_metric_dep,
        device="cpu",
        config=pipeline_config,
        checkpoint_manager=ckpt_manager,
        early_stopping=early_stopping,
        run_dir=tmp_path,
    )

    # Train for 2 epochs
    results = trainer.fit(epochs=2, start_epoch=1)
    assert len(results["history"]) == 2
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "history.json").exists()
    assert (tmp_path / "config.yaml").exists()

    # Resume for 1 more epoch (epoch 3)
    model_resume = create_model("baseline_cnn")
    opt_resume = create_optimizer(
        model_resume.parameters(), OptimizerConfig(name="adamw", learning_rate=0.001)
    )
    sched_resume, is_metric_dep2 = create_scheduler(
        opt_resume, SchedulerConfig(name="reduce_on_plateau", patience=1)
    )

    checkpoint = CheckpointManager.load(
        checkpoint_path=tmp_path / "last.pt",
        model=model_resume,
        optimizer=opt_resume,
        scheduler=sched_resume,
    )
    assert checkpoint["epoch"] == 2

    trainer_resume = Trainer(
        model=model_resume,
        optimizer=opt_resume,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=sched_resume,
        is_scheduler_metric_dependent=is_metric_dep2,
        device="cpu",
        config=pipeline_config,
        checkpoint_manager=ckpt_manager,
        run_dir=tmp_path,
    )
    trainer_resume.history = list(checkpoint["history"])

    resume_results = trainer_resume.fit(epochs=3, start_epoch=3)
    assert len(resume_results["history"]) == 3
    assert resume_results["history"][-1]["epoch"] == 3


def test_real_dataset_one_epoch_integration(tmp_path: Path) -> None:
    """Verify integration with real Phase 03 DataLoaders."""
    raw_csv = ROOT_DIR / "data/raw/fer2013/fer2013.csv"
    if not raw_csv.exists():
        pytest.skip("FER2013 dataset not downloaded on this machine")

    train_loader, val_loader, _ = build_dataloaders(data_path=raw_csv, batch_size=32)

    model = create_model("baseline_cnn")
    optimizer = create_optimizer(
        model.parameters(), OptimizerConfig(name="adamw", learning_rate=0.001)
    )
    criterion = create_loss("cross_entropy")

    # Take a 1-batch subset for quick deterministic smoke test
    real_train_batch = next(iter(train_loader))
    real_val_batch = next(iter(val_loader))

    mini_train_loader = [real_train_batch]
    mini_val_loader = [real_val_batch]

    ckpt_manager = CheckpointManager(save_dir=tmp_path)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=mini_train_loader,  # type: ignore[arg-type]
        val_loader=mini_val_loader,  # type: ignore[arg-type]
        device="cpu",
        checkpoint_manager=ckpt_manager,
        run_dir=tmp_path,
    )

    results = trainer.fit(epochs=1, start_epoch=1)
    assert len(results["history"]) == 1
    assert results["history"][0]["train_loss"] > 0.0
    assert 0.0 <= results["history"][0]["train_accuracy"] <= 1.0
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()
