"""Unit tests for Champion model loading, device resolution, and warmup."""

from __future__ import annotations

import pytest
import torch

from ml.inference.config import ModelInferenceConfig
from ml.inference.model_loader import ModelManager, load_champion_model, resolve_device


def test_resolve_device():
    """Verify device resolution for cpu and auto."""
    dev_cpu = resolve_device("cpu")
    assert dev_cpu.type == "cpu"

    dev_auto = resolve_device("auto")
    assert dev_auto.type in ("cpu", "cuda")

    with pytest.raises(ValueError):
        resolve_device("tpu_device")


def test_model_manager_load_and_caching():
    """Verify Champion model loading, metadata extraction, and caching."""
    cfg = ModelInferenceConfig(
        champion_dir="artifacts/optimized/champion",
        metadata_file="artifacts/optimized/champion/metadata.json",
        weights_file="artifacts/optimized/champion/model.pt",
    )
    manager = ModelManager(config=cfg, device=torch.device("cpu"))
    model, metadata = manager.load_champion()

    assert model is not None
    assert metadata["model_name"] == "champion-pruning-30"
    assert not model.training  # In eval mode

    # Second call returns cached instance
    m2, _ = manager.load_champion()
    assert m2 is model

    # Forward pass test
    dummy = torch.randn(2, 1, 48, 48)
    with torch.inference_mode():
        out = model(dummy)
    assert out.shape == (2, 7)
    assert not torch.isnan(out).any()


def test_model_warmup_immutability():
    """Verify warm-up execution does not alter model parameters."""
    model, _, _ = load_champion_model(warmup_iterations=0)

    # Save parameter snapshot before warmup
    weights_before = [p.detach().clone() for p in model.parameters()]

    manager = ModelManager()
    manager.model = model
    manager.warmup(iterations=3)

    # Compare parameter snapshot after warmup
    for p_before, p_after in zip(weights_before, model.parameters(), strict=True):
        assert torch.equal(p_before, p_after)
