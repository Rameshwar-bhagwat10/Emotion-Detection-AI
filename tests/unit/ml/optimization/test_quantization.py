"""Unit tests for Post-Training Quantization (INT8 PTQ)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.models.factory import create_model
from ml.optimization.quantization import calibrate_static_ptq, quantize_model_dynamic


def test_quantize_model_dynamic_forward() -> None:
    """Verify dynamic INT8 quantized model produces valid [B, 7] logits."""
    model = create_model("baseline_cnn")
    quantized = quantize_model_dynamic(model)
    quantized.eval()

    dummy = torch.randn(4, 1, 48, 48, dtype=torch.float32)
    with torch.no_grad():
        out = quantized(dummy)

    assert out.shape == (4, 7)
    assert torch.isfinite(out).all()


def test_calibrate_static_ptq_pipeline() -> None:
    """Verify PTQ calibration pipeline runs on synthetic data."""
    model = create_model("baseline_cnn")
    x = torch.randn(16, 1, 48, 48)
    y = torch.randint(0, 7, (16,))
    ds = TensorDataset(x, y)
    loader = DataLoader(
        ds,
        batch_size=4,
        collate_fn=lambda batch: {
            "image": torch.stack([item[0] for item in batch]),
            "label": torch.stack([item[1] for item in batch]),
        },
    )

    quantized = calibrate_static_ptq(model, loader, num_samples=16)
    quantized.eval()

    with torch.no_grad():
        out = quantized(x[:2])
    assert out.shape == (2, 7)
