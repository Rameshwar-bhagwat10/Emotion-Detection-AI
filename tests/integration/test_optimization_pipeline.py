"""Integration tests for Phase 08 optimization pipeline."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.models.factory import create_model
from ml.optimization.distillation import DistillationConfig, train_student_distillation
from ml.optimization.exporter import export_to_onnx, validate_onnx_export
from ml.optimization.pruning import (
    apply_magnitude_pruning,
    finalize_pruning,
    fine_tune_pruned_model,
)
from ml.optimization.quantization import quantize_model_dynamic


def create_synthetic_dataloader(num_samples: int = 16, batch_size: int = 4) -> DataLoader:
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


def test_optimization_pipeline_end_to_end(tmp_path: Path) -> None:
    """Verify entire optimization cycle runs cleanly on synthetic data."""
    train_loader = create_synthetic_dataloader(num_samples=16)
    val_loader = create_synthetic_dataloader(num_samples=16)

    # 1. Base Model
    model = create_model("baseline_cnn")

    # 2. INT8 Quantization
    int8_model = quantize_model_dynamic(model)
    dummy = torch.randn(2, 1, 48, 48)
    with torch.no_grad():
        out_int8 = int8_model(dummy)
    assert out_int8.shape == (2, 7)

    # 3. Pruning + Fine-tuning
    pruned = apply_magnitude_pruning(model, amount=0.10)
    pruned_ft, history = fine_tune_pruned_model(
        pruned,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        learning_rate=0.001,
        device="cpu",
    )
    finalized = finalize_pruning(pruned_ft)
    assert len(history["train_loss"]) == 1

    # 4. Distillation
    student = create_model("baseline_cnn")
    cfg = DistillationConfig(epochs=1, learning_rate=0.001)
    trained_student, dist_history = train_student_distillation(
        teacher=model,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device="cpu",
    )
    assert len(dist_history["train_loss"]) == 1

    # 5. Export to ONNX & Validate
    onnx_path = tmp_path / "final_model.onnx"
    export_to_onnx(finalized, onnx_path)
    val_report = validate_onnx_export(onnx_path, finalized, atol=1e-3, num_samples=2)
    assert val_report["is_valid"] is True
