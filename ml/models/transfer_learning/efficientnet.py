"""EfficientNet-B0 transfer learning architecture for facial expression recognition."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from ml.models.transfer_learning.config import TransferLearningConfig


class EfficientNetB0Transfer(nn.Module):
    """EfficientNet-B0 transfer learning model with grayscale adaptation and custom classification head."""

    def __init__(
        self,
        config: TransferLearningConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize EfficientNet-B0 Transfer Learning model.

        Args:
            config: Optional TransferLearningConfig instance or dictionary.
            **kwargs: Overrides for configuration arguments.
        """
        super().__init__()

        if config is None:
            self.config = TransferLearningConfig(model_name="efficientnet_b0", **kwargs)
        elif isinstance(config, dict):
            merged = {**config, **kwargs}
            self.config = TransferLearningConfig.from_dict(merged)
        else:
            self.config = config

        weights = EfficientNet_B0_Weights.DEFAULT if self.config.pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features  # 1280
        dropout = self.config.dropout_rate if self.config.dropout_rate > 0.0 else 0.2
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, self.config.num_classes),
        )

        if self.config.freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W] (e.g. [B, 1, 48, 48] or [B, 3, 112, 112]).

        Returns:
            Raw unnormalized logits of shape [B, num_classes].
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor [B, C, H, W], got shape {list(x.shape)}")
        if x.shape[1] != self.config.in_channels:
            raise ValueError(f"Expected {self.config.in_channels} input channels, got {x.shape[1]}")

        # Grayscale adaptation
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        logits: torch.Tensor = self.backbone(x)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all feature extraction layers."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
