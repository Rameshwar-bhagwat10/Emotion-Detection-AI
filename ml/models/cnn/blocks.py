"""Reusable Convolutional and Neural Network Blocks for CNN architectures."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Reusable Convolutional Block: Conv2d -> BatchNorm2d -> ReLU -> Optional MaxPool2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool: bool = True,
        pool_kernel: int = 2,
        pool_stride: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_pool = pool

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,  # BatchNorm follows immediately
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)

        if activation.lower() == "relu":
            self.act: nn.Module = nn.ReLU(inplace=True)
        elif activation.lower() == "leaky_relu":
            self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif activation.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(
                f"Unsupported activation '{activation}'. Expected 'relu', 'leaky_relu', or 'gelu'."
            )

        if pool:
            self.pool: nn.Module = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        else:
            self.pool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Conv2d -> BatchNorm2d -> Activation -> Pool."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x

    def extra_repr(self) -> str:
        """String representation of block configuration."""
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"pool={self.has_pool}"
        )
