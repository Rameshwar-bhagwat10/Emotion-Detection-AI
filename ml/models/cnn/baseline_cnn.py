"""Baseline CNN Architecture for 7-class Facial Expression Recognition (FER2013)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ml.datasets.fer2013.parser import EMOTION_NAMES
from ml.models.cnn.blocks import ConvBlock
from ml.models.cnn.config import BaselineCNNConfig


class BaselineCNN(nn.Module):
    """4-stage Convolutional Neural Network baseline for facial expression emotion classification.

    Input Contract:
        torch.Tensor of shape [B, 1, 48, 48] and dtype torch.float32

    Output Contract:
        torch.Tensor of shape [B, 7] representing raw class logits (NO Softmax applied)
    """

    def __init__(
        self,
        config: BaselineCNNConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Resolve configuration
        if config is None:
            self.config = BaselineCNNConfig(**kwargs) if kwargs else BaselineCNNConfig.from_yaml()
        elif isinstance(config, dict):
            merged = dict(config)
            merged.update(kwargs)
            self.config = BaselineCNNConfig.from_dict(merged)
        else:
            self.config = config

        self.input_channels = self.config.input_channels
        self.input_size = self.config.input_size
        self.num_classes = self.config.num_classes
        self.class_names = list(EMOTION_NAMES)

        # 1. Feature Extractor (Convolutional Stages)
        stages_list: list[nn.Module] = []
        for stage in self.config.stages:
            stages_list.append(
                ConvBlock(
                    in_channels=stage.in_channels,
                    out_channels=stage.out_channels,
                    kernel_size=stage.kernel_size,
                    stride=stage.stride,
                    padding=stage.padding,
                    pool=stage.pool,
                    pool_kernel=stage.pool_kernel,
                    pool_stride=stage.pool_stride,
                    activation=self.config.activation,
                )
            )
        self.features = nn.Sequential(*stages_list)

        # 2. Global Spatial Pooling & Flattening
        self.pool = nn.AdaptiveAvgPool2d(self.config.adaptive_pool_size)
        self.flatten = nn.Flatten()

        # Last stage output channels
        last_stage_channels = self.config.stages[-1].out_channels

        # 3. Classifier Head (Linear -> ReLU -> Dropout -> Linear)
        self.classifier = nn.Sequential(
            nn.Linear(last_stage_channels, self.config.classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(self.config.classifier_hidden, self.num_classes),
        )

        # Initialize network weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming Normal (He) initialization for Conv/Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor, pooling, and classification head.

        Args:
            x: Input tensor of shape [B, 1, 48, 48], dtype float32.

        Returns:
            Raw unnormalized logits tensor of shape [B, 7].
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor [B, C, H, W], got {x.ndim}D tensor of shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Input channel mismatch: expected {self.input_channels} channel(s), got {x.shape[1]} channel(s) in shape {tuple(x.shape)}"
            )

        # Feature extraction: [B, 1, 48, 48] -> [B, 256, 3, 3]
        feat = self.features(x)

        # Adaptive pooling: [B, 256, 3, 3] -> [B, 256, 1, 1]
        pooled = self.pool(feat)

        # Flatten: [B, 256, 1, 1] -> [B, 256]
        flat = self.flatten(pooled)

        # Classification logits: [B, 256] -> [B, 7]
        logits: torch.Tensor = self.classifier(flat)
        return logits

    def get_parameter_count(self) -> dict[str, int]:
        """Calculate total, trainable, and non-trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": non_trainable,
        }

    def get_model_summary(self) -> dict[str, Any]:
        """Return structured model metadata and summary."""
        params = self.get_parameter_count()
        return {
            "model_name": self.config.name,
            "model_version": self.config.version,
            "description": self.config.description,
            "input_shape": (1, self.input_channels, self.input_size[0], self.input_size[1]),
            "output_shape": (1, self.num_classes),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "parameters": params,
        }

    def trace_shapes(
        self, input_shape: tuple[int, ...] = (1, 1, 48, 48)
    ) -> list[tuple[str, tuple[int, ...]]]:
        """Trace output tensor dimensions across each major block of the model."""
        traces: list[tuple[str, tuple[int, ...]]] = [("input", input_shape)]
        x = torch.zeros(input_shape, dtype=torch.float32)

        for i, stage in enumerate(self.features):
            x = stage(x)
            traces.append((f"stage_{i + 1}_{stage.__class__.__name__}", tuple(x.shape)))

        x = self.pool(x)
        traces.append(("adaptive_pool", tuple(x.shape)))

        x = self.flatten(x)
        traces.append(("flatten", tuple(x.shape)))

        for i, layer in enumerate(self.classifier):
            x = layer(x)
            traces.append((f"classifier_{i + 1}_{layer.__class__.__name__}", tuple(x.shape)))

        return traces
