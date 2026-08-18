"""Unit tests for reduced precision and numerical divergence evaluation."""

from __future__ import annotations

import torch

from ml.models.factory import create_model
from ml.optimization.precision import ReducedPrecisionWrapper, evaluate_numerical_difference


def test_reduced_precision_wrapper_forward() -> None:
    """Verify ReducedPrecisionWrapper handles [B, 1, 48, 48] and returns [B, 7] finite outputs."""
    model = create_model("baseline_cnn")
    wrapper = ReducedPrecisionWrapper(model, dtype=torch.float32)
    wrapper.eval()

    dummy = torch.randn(4, 1, 48, 48, dtype=torch.float32)
    with torch.no_grad():
        out = wrapper(dummy)

    assert out.shape == (4, 7)
    assert torch.isfinite(out).all()


def test_evaluate_numerical_difference() -> None:
    """Verify numerical divergence computation between models."""
    model_a = create_model("baseline_cnn")
    model_b = create_model("baseline_cnn")

    dummy = torch.randn(2, 1, 48, 48, dtype=torch.float32)
    diff = evaluate_numerical_difference(model_a, model_b, dummy)

    assert "max_absolute_error" in diff
    assert "mean_absolute_error" in diff
    assert "cosine_similarity" in diff
    assert diff["max_absolute_error"] >= 0.0
