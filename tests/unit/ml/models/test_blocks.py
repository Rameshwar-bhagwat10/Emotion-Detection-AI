"""Unit tests for reusable CNN Blocks."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ml.models.cnn.blocks import ConvBlock


def test_conv_block_initialization_and_forward() -> None:
    """Verify ConvBlock forward pass preserves correct dimensions with pooling."""
    block = ConvBlock(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, pool=True)
    x = torch.randn(2, 1, 48, 48, dtype=torch.float32)

    out = block(x)

    assert isinstance(out, torch.Tensor)
    # 48x48 -> Conv 48x48 -> MaxPool(2,2) -> 24x24
    assert out.shape == (2, 32, 24, 24)
    assert out.dtype == torch.float32


def test_conv_block_without_pooling() -> None:
    """Verify ConvBlock without pooling preserves spatial dimensions."""
    block = ConvBlock(
        in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, pool=False
    )
    x = torch.randn(2, 32, 24, 24, dtype=torch.float32)

    out = block(x)

    assert out.shape == (2, 64, 24, 24)
    assert isinstance(block.pool, nn.Identity)


def test_conv_block_activations() -> None:
    """Verify ConvBlock supports supported activation functions."""
    b_relu = ConvBlock(1, 16, activation="relu")
    b_leaky = ConvBlock(1, 16, activation="leaky_relu")
    b_gelu = ConvBlock(1, 16, activation="gelu")

    assert isinstance(b_relu.act, nn.ReLU)
    assert isinstance(b_leaky.act, nn.LeakyReLU)
    assert isinstance(b_gelu.act, nn.GELU)

    with pytest.raises(ValueError, match="Unsupported activation"):
        ConvBlock(1, 16, activation="invalid_activation")
