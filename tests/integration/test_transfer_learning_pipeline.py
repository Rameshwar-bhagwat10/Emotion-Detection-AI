"""Integration tests for Phase 07 transfer learning and model selection pipeline."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.evaluation.evaluator import Evaluator
from ml.models.factory import create_model
from ml.training.trainer import Trainer

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def create_synthetic_dataloader(batch_size: int = 4, num_samples: int = 16) -> DataLoader:
    """Create synthetic 1-channel image dataloader."""
    x = torch.randn(num_samples, 1, 48, 48, dtype=torch.float32)
    y = torch.randint(0, 7, (num_samples,), dtype=torch.long)
    ds = TensorDataset(x, y)

    def collate_fn(batch):
        return {
            "image": torch.stack([item[0] for item in batch]),
            "label": torch.stack([item[1] for item in batch]),
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def test_resnet18_trainer_one_epoch(tmp_path: Path) -> None:
    """Verify ResNet-18 model runs cleanly in Phase 05 Trainer pipeline."""
    model = create_model("resnet18", pretrained=False)
    train_loader = create_synthetic_dataloader(batch_size=4, num_samples=8)
    val_loader = create_synthetic_dataloader(batch_size=4, num_samples=8)

    from ml.training.config import OptimizerConfig
    from ml.training.losses import create_loss
    from ml.training.optimizers import create_optimizer

    optimizer = create_optimizer(
        model.parameters(), OptimizerConfig(name="adamw", learning_rate=0.001)
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


def test_mobilenet_v3_small_evaluator_integration(tmp_path: Path) -> None:
    """Verify MobileNetV3-Small evaluates cleanly with Phase 06 Evaluator."""
    model = create_model("mobilenet_v3_small", pretrained=False)
    test_loader = create_synthetic_dataloader(batch_size=4, num_samples=8)

    evaluator = Evaluator(
        model=model,
        data_loader=test_loader,
        device="cpu",
        run_dir=tmp_path / "mobilenet_eval",
    )

    results = evaluator.evaluate()
    assert "summary" in results
    assert results["summary"]["total_samples"] == 8
    assert (tmp_path / "mobilenet_eval" / "evaluation_results.json").exists()
