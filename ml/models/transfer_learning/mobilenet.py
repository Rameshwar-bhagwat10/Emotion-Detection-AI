"""MobileNetV3-Small transfer learning architecture for facial expression recognition."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from ml.models.transfer_learning.config import TransferLearningConfig


class MobileNetV3SmallTransfer(nn.Module):
    """MobileNetV3-Small transfer learning model with grayscale input adaptation and custom head."""

    def __init__(
        self,
        config: TransferLearningConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize MobileNetV3-Small Transfer Learning model.

        Args:
            config: Optional TransferLearningConfig instance or dictionary.
            **kwargs: Overrides for configuration arguments.
        """
        super().__init__()

        if config is None:
            self.config = TransferLearningConfig(model_name="mobilenet_v3_small", **kwargs)
        elif isinstance(config, dict):
            merged = {**config, **kwargs}
            self.config = TransferLearningConfig.from_dict(merged)
        else:
            self.config = config

        # Load pretrained or randomly initialized MobileNetV3-Small
        weights = MobileNet_V3_Small_Weights.DEFAULT if self.config.pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        # Replace final classifier layer (classifier is Sequential(Linear, Hardswish, Dropout, Linear))
        in_features = self.backbone.classifier[3].in_features  # 1024
        if self.config.dropout_rate > 0.0:
            self.backbone.classifier[2] = nn.Dropout(p=self.config.dropout_rate)
        self.backbone.classifier[3] = nn.Linear(in_features, self.config.num_classes)

        if self.config.freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W] (e.g. [B, 1, 48, 48]).

        Returns:
            Raw unnormalized logits of shape [B, num_classes].
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor [B, C, H, W], got shape {list(x.shape)}")
        if x.shape[1] != self.config.in_channels:
            raise ValueError(f"Expected {self.config.in_channels} input channels, got {x.shape[1]}")

        # Grayscale adaptation: repeat channel 3 times for standard 3-channel vision backbone
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        logits: torch.Tensor = self.backbone(x)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze feature extraction layers."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all model parameters for full end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_backbone_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to the feature extraction backbone."""
        return list(self.backbone.features.parameters())

    def get_head_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to the classifier head."""
        return list(self.backbone.classifier.parameters())
