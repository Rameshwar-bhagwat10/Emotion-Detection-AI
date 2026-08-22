"""ResNet-18 transfer learning architecture for facial expression recognition with optional SE attention."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from ml.models.transfer_learning.config import TransferLearningConfig


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weight = self.fc(x).view(b, c, 1, 1)
        return x * weight


class ResNet18Transfer(nn.Module):
    """ResNet-18 transfer learning model with grayscale input adaptation, optional SE attention, and custom head."""

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

        # Optional Squeeze-and-Excitation attention before global pooling
        self.use_attention = getattr(self.config, "use_attention", False)
        if self.use_attention:
            self.se_block = SEBlock(channels=512, reduction=16)
        else:
            self.se_block = None

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
            x: Input tensor of shape [B, C, H, W] (e.g. [B, 1, 48, 48] or [B, 3, 112, 112]).

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

        if self.use_attention and self.se_block is not None:
            # Forward through convolutional stages up to layer4
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

            # Apply SE channel attention
            x = self.se_block(x)

            # Global average pool and fc
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            logits: torch.Tensor = self.backbone.fc(x)
            return logits

        logits = self.backbone(x)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all layers except the classification head."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc") and not name.startswith("se_block"):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all model parameters for full end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_backbone_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to the feature extraction backbone."""
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.fc")]

    def get_head_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to the classification head."""
        return [p for n, p in self.named_parameters() if n.startswith("backbone.fc")]
