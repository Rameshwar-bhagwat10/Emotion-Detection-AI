"""Unit tests for magnitude pruning, sparsity measurement, and fine-tuning."""

from __future__ import annotations

import torch

from ml.models.factory import create_model
from ml.optimization.pruning import (
    apply_magnitude_pruning,
    compute_model_sparsity,
    finalize_pruning,
)


def test_apply_magnitude_pruning_and_sparsity() -> None:
    """Verify magnitude pruning introduces sparsity and compute_model_sparsity accurately measures it."""
    model = create_model("baseline_cnn")
    initial_sparsity = compute_model_sparsity(model)
    assert initial_sparsity["sparsity_percentage"] < 5.0

    pruned = apply_magnitude_pruning(model, amount=0.20)
    pruned_sparsity = compute_model_sparsity(pruned)
    assert pruned_sparsity["sparsity_percentage"] > 10.0

    finalized = finalize_pruning(pruned)
    finalized_sparsity = compute_model_sparsity(finalized)
    assert finalized_sparsity["sparsity_percentage"] > 10.0

    # Ensure model forward still works
    dummy = torch.randn(2, 1, 48, 48)
    with torch.no_grad():
        out = finalized(dummy)
    assert out.shape == (2, 7)
