"""Unit tests for ResNet-18 and MobileNetV3-Small transfer learning architectures."""

from __future__ import annotations

import pytest
import torch

from ml.models.factory import create_model
from ml.models.transfer_learning.config import TransferLearningConfig
from ml.models.transfer_learning.mobilenet import MobileNetV3SmallTransfer
from ml.models.transfer_learning.resnet import ResNet18Transfer


def test_resnet18_transfer_instantiation_and_forward() -> None:
    """Verify ResNet-18 model instantiates, handles 1-channel grayscale, and produces [B, 7] logits."""
    config = TransferLearningConfig(
        model_name="resnet18", num_classes=7, in_channels=1, pretrained=False
    )
    model = ResNet18Transfer(config=config)
    model.eval()

    dummy = torch.randn(4, 1, 48, 48, dtype=torch.float32)
    with torch.no_grad():
        logits = model(dummy)

    assert logits.shape == (4, 7)
    assert torch.isfinite(logits).all()


def test_resnet18_transfer_gradient_flow_and_freeze() -> None:
    """Verify ResNet-18 gradients flow properly and backbone freezing works."""
    model = ResNet18Transfer(pretrained=False)
    model.train()

    # Freeze backbone
    model.freeze_backbone()
    dummy = torch.randn(2, 1, 48, 48, dtype=torch.float32)
    logits = model(dummy)
    loss = logits.sum()
    loss.backward()

    # Backbone weights should NOT have gradients, head should have gradients
    for p in model.get_backbone_parameters():
        assert p.grad is None or not p.requires_grad

    for p in model.get_head_parameters():
        assert p.requires_grad
        assert p.grad is not None


def test_mobilenet_v3_small_transfer_instantiation_and_forward() -> None:
    """Verify MobileNetV3-Small model instantiates and produces [B, 7] logits."""
    config = TransferLearningConfig(
        model_name="mobilenet_v3_small", num_classes=7, in_channels=1, pretrained=False
    )
    model = MobileNetV3SmallTransfer(config=config)
    model.eval()

    dummy = torch.randn(4, 1, 48, 48, dtype=torch.float32)
    with torch.no_grad():
        logits = model(dummy)

    assert logits.shape == (4, 7)
    assert torch.isfinite(logits).all()


def test_mobilenet_v3_small_freeze_and_unfreeze() -> None:
    """Verify MobileNetV3-Small backbone freeze and unfreeze toggles."""
    model = MobileNetV3SmallTransfer(pretrained=False)

    model.freeze_backbone()
    for p in model.get_backbone_parameters():
        assert not p.requires_grad

    model.unfreeze_backbone()
    for p in model.get_backbone_parameters():
        assert p.requires_grad


def test_model_factory_creates_transfer_models() -> None:
    """Verify model factory instantiates resnet18 and mobilenet_v3_small."""
    res_model = create_model("resnet18", pretrained=False)
    assert isinstance(res_model, ResNet18Transfer)

    mob_model = create_model("mobilenet_v3_small", pretrained=False)
    assert isinstance(mob_model, MobileNetV3SmallTransfer)


def test_transfer_config_validation() -> None:
    """Verify validation error on invalid configuration."""
    with pytest.raises(ValueError, match="num_classes must be positive"):
        TransferLearningConfig(num_classes=0)

    with pytest.raises(ValueError, match="dropout_rate must be in"):
        TransferLearningConfig(dropout_rate=1.5)
