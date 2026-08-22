"""Model architectures and attention-enhanced variants for Model V2."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    convnext_tiny,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)

from ml.models.layers.attention import CBAMBlock, SEBlock


class ResNetV2Transfer(nn.Module):
    """ResNet Transfer Learning model with grayscale/RGB support, custom attention, and calibrated head."""

    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 7,
        pretrained: bool = True,
        in_channels: int = 1,
        dropout_rate: float = 0.2,
        attention_type: str | None = None,  # None, 'se', 'cbam'
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower().strip()
        self.in_channels = in_channels
        self.attention_type = attention_type.lower().strip() if attention_type else None

        if self.architecture == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
            feature_dim = 2048
        else:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
            feature_dim = 512

        # Attention layer before global pooling
        if self.attention_type == "se":
            self.attention = SEBlock(channels=feature_dim, reduction=16)
        elif self.attention_type == "cbam":
            self.attention = CBAMBlock(in_planes=feature_dim, ratio=16)
        else:
            self.attention = None

        # Replace classification head
        if dropout_rate > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(feature_dim, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if self.attention is not None:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

            x = self.attention(x)

            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            logits: torch.Tensor = self.backbone.fc(x)
            return logits

        return self.backbone(x)


class EfficientNetV2Transfer(nn.Module):
    """EfficientNet-B0 model for Model V2 benchmarking."""

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        in_channels: int = 1,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        try:
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
        except Exception:
            self.backbone = efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


class MobileNetV3Transfer(nn.Module):
    """MobileNetV3-Small for lightweight real-time benchmark."""

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        in_channels: int = 1,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
        except Exception:
            self.backbone = mobilenet_v3_small(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


class ConvNeXtTinyTransfer(nn.Module):
    """ConvNeXt-Tiny modern convolutional architecture."""

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        in_channels: int = 1,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        try:
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.backbone = convnext_tiny(weights=weights)
        except Exception:
            self.backbone = convnext_tiny(weights=None)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


def create_v2_model(
    name: str = "resnet18_cbam",
    num_classes: int = 7,
    pretrained: bool = True,
    in_channels: int = 1,
    dropout_rate: float = 0.2,
) -> nn.Module:
    """Factory function for creating V2 candidate models."""
    name_lower = name.lower().strip()

    if name_lower in ("resnet18", "resnet-18", "resnet18_v1"):
        return ResNetV2Transfer("resnet18", num_classes, pretrained, in_channels, dropout_rate, None)
    elif name_lower in ("resnet18_se", "se_resnet18"):
        return ResNetV2Transfer("resnet18", num_classes, pretrained, in_channels, dropout_rate, "se")
    elif name_lower in ("resnet18_cbam", "cbam_resnet18"):
        return ResNetV2Transfer("resnet18", num_classes, pretrained, in_channels, dropout_rate, "cbam")
    elif name_lower in ("resnet50", "resnet-50"):
        return ResNetV2Transfer("resnet50", num_classes, pretrained, in_channels, dropout_rate, None)
    elif name_lower in ("resnet50_cbam", "cbam_resnet50"):
        return ResNetV2Transfer("resnet50", num_classes, pretrained, in_channels, dropout_rate, "cbam")
    elif name_lower in ("efficientnet_b0", "efficientnet-b0"):
        return EfficientNetV2Transfer(num_classes, pretrained, in_channels, dropout_rate)
    elif name_lower in ("mobilenet_v3_small", "mobilenetv3"):
        return MobileNetV3Transfer(num_classes, pretrained, in_channels, dropout_rate)
    elif name_lower in ("convnext_tiny", "convnext"):
        return ConvNeXtTinyTransfer(num_classes, pretrained, in_channels, dropout_rate)
    else:
        raise ValueError(f"Unknown model name '{name}' for V2.")
