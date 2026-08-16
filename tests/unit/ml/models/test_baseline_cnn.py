"""Unit tests for BaselineCNN architecture and configuration."""

from __future__ import annotations

import pytest
import torch

from ml.models.cnn.baseline_cnn import BaselineCNN
from ml.models.cnn.config import BaselineCNNConfig, StageConfig


def test_baseline_cnn_config_validation() -> None:
    """Verify BaselineCNNConfig schema validation."""
    # 1. Valid defaults
    cfg = BaselineCNNConfig()
    assert cfg.input_channels == 1
    assert cfg.num_classes == 7
    assert len(cfg.stages) == 4

    # 2. Invalid input_channels
    with pytest.raises(ValueError, match="input_channels must be > 0"):
        BaselineCNNConfig(input_channels=0)

    # 3. Invalid num_classes
    with pytest.raises(ValueError, match="num_classes must be > 0"):
        BaselineCNNConfig(num_classes=-1)

    # 4. Invalid dropout
    with pytest.raises(ValueError, match="dropout must be in range"):
        BaselineCNNConfig(dropout=1.5)

    # 5. Invalid stage connectivity
    invalid_stages = [
        StageConfig(in_channels=1, out_channels=32),
        StageConfig(in_channels=64, out_channels=128),  # 32 != 64 mismatch
    ]
    with pytest.raises(ValueError, match="does not match Stage 1 in_channels"):
        BaselineCNNConfig(stages=invalid_stages)


def test_baseline_cnn_forward_shape_and_dtype() -> None:
    """Verify forward pass returns [B, 7] float32 raw logits."""
    model = BaselineCNN()
    model.eval()

    dummy_input = torch.randn(4, 1, 48, 48, dtype=torch.float32)
    with torch.no_grad():
        logits = model(dummy_input)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (4, 7)
    assert logits.dtype == torch.float32
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_baseline_cnn_input_channel_and_dimension_validation() -> None:
    """Verify model raises ValueError on invalid input shapes or channel counts."""
    model = BaselineCNN()

    # 1. Wrong channel count (e.g. 3 channels)
    rgb_input = torch.randn(2, 3, 48, 48, dtype=torch.float32)
    with pytest.raises(ValueError, match="Input channel mismatch"):
        model(rgb_input)

    # 2. Wrong dimensions (e.g. 3D tensor instead of 4D batch)
    dim3_input = torch.randn(1, 48, 48, dtype=torch.float32)
    with pytest.raises(ValueError, match="Expected 4D input tensor"):
        model(dim3_input)


def test_baseline_cnn_parameter_count_and_weights() -> None:
    """Verify parameter count calculations and finite initialized weights."""
    model = BaselineCNN()
    counts = model.get_parameter_count()

    assert counts["total"] > 0
    assert counts["trainable"] == counts["total"]
    assert counts["non_trainable"] == 0

    # Ensure no NaN / Inf in initialized parameters
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"Parameter {name} contains NaN"
        assert not torch.isinf(param).any(), f"Parameter {name} contains Inf"


def test_baseline_cnn_gradient_flow() -> None:
    """Verify network is fully differentiable and receives backward gradients."""
    model = BaselineCNN()
    model.train()
    model.zero_grad()

    x = torch.randn(2, 1, 48, 48, dtype=torch.float32, requires_grad=True)
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients"
            assert not torch.isnan(param.grad).any()


def test_baseline_cnn_train_eval_modes() -> None:
    """Verify model.train() and model.eval() toggle submodules correctly."""
    model = BaselineCNN()

    model.train()
    assert model.training
    assert model.classifier[2].training  # Dropout module

    model.eval()
    assert not model.training
    assert not model.classifier[2].training  # Dropout module


def test_baseline_cnn_state_dict_serialization() -> None:
    """Verify state_dict save/load consistency produces bit-identical outputs."""
    model_a = BaselineCNN()
    model_a.eval()

    state = model_a.state_dict()
    model_b = BaselineCNN()
    model_b.load_state_dict(state)
    model_b.eval()

    test_input = torch.randn(4, 1, 48, 48, dtype=torch.float32)
    with torch.no_grad():
        out_a = model_a(test_input)
        out_b = model_b(test_input)

    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_baseline_cnn_trace_shapes() -> None:
    """Verify trace_shapes outputs valid layer dimensions."""
    model = BaselineCNN()
    traces = model.trace_shapes((1, 1, 48, 48))

    assert len(traces) > 0
    assert traces[0] == ("input", (1, 1, 48, 48))
    # Last shape should be [1, 7]
    assert traces[-1][1] == (1, 7)
