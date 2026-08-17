"""ResNet-18 transfer learning architecture for facial expression recognition."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from ml.models.transfer_learning.config import TransferLearningConfig


class ResNet18Transfer(nn.Module):
    """ResNet-18 transfer learning model with grayscale input adaptation and custom head."""

    def __init__(
        self,
        config: TransferLearningConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ResNet-18 Transfer Learning model.

        Args:
            config: Optional TransferLearningConfig instance or dictionary.
            **kwargs: Overrides for configuration arguments.
        """
        super().__init__()

        if config is None:
            self.config = TransferLearningConfig(model_name="resnet18", **kwargs)
        elif isinstance(config, dict):
            merged = {**config, **kwargs}
            self.config = TransferLearningConfig.from_dict(merged)
        else:
            self.config = config

        # Load pretrained or randomly initialized ResNet-18
        weights = ResNet18_Weights.DEFAULT if self.config.pretrained else None
        self.backbone = resnet18(weights=weights)

        # Replace classification head
        in_features = self.backbone.fc.in_features  # 512
        if self.config.dropout_rate > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=self.config.dropout_rate),
                nn.Linear(in_features, self.config.num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, self.config.num_classes)

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
        """Freeze all layers except the classification head."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all model parameters for full end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_backbone_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to the feature extraction backbone."""
        return [p for n, p in self.backbone.named_parameters() if not n.startswith("fc")]

    def get_head_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to the classification head."""
        return [p for n, p in self.backbone.named_parameters() if n.startswith("fc")]
