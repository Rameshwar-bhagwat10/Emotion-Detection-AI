"""Model Factory for instantiating model architectures from configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from ml.models.cnn.baseline_cnn import BaselineCNN
from ml.models.cnn.config import BaselineCNNConfig
from ml.models.registry import get_model_class, register_model
from ml.models.transfer_learning.config import TransferLearningConfig
from ml.models.transfer_learning.efficientnet import EfficientNetB0Transfer
from ml.models.transfer_learning.mobilenet import MobileNetV3SmallTransfer
from ml.models.transfer_learning.models_v2 import (
    ConvNeXtTinyTransfer,
    EfficientNetV2Transfer,
    MobileNetV3Transfer,
    ResNetV2Transfer,
)
from ml.models.transfer_learning.resnet import ResNet18Transfer

# Register known architectures
register_model("baseline_cnn")(BaselineCNN)
register_model("baseline_cnn_v1")(BaselineCNN)
register_model("resnet18")(ResNet18Transfer)
register_model("resnet18_v1")(ResNet18Transfer)
register_model("resnet18_se")(ResNet18Transfer)
register_model("resnet18_cbam")(lambda **kwargs: ResNetV2Transfer(architecture="resnet18", attention_type="cbam", **kwargs))
register_model("resnet18_v2")(lambda **kwargs: ResNetV2Transfer(architecture="resnet18", attention_type="cbam", **kwargs))
register_model("mobilenet_v3_small")(MobileNetV3SmallTransfer)
register_model("mobilenet_v3_small_v1")(MobileNetV3SmallTransfer)
register_model("efficientnet_b0")(EfficientNetB0Transfer)
register_model("efficientnet_b0_v1")(EfficientNetB0Transfer)
register_model("convnext_tiny")(ConvNeXtTinyTransfer)


def create_model(
    model_name: str = "baseline_cnn",
    config: BaselineCNNConfig | TransferLearningConfig | dict[str, Any] | None = None,
    config_path: str | Path = "ml/configs/models.yaml",
    **kwargs: Any,
) -> nn.Module:
    """Create and initialize a model instance based on name and configuration.

    Args:
        model_name: Identifier of the model to create (e.g. 'baseline_cnn', 'resnet18', 'mobilenet_v3_small').
        config: Optional configuration object or dictionary.
        config_path: Path to models.yaml configuration file if config is not provided.
        **kwargs: Overrides for configuration arguments.

    Returns:
        Instantiated PyTorch nn.Module.

    Raises:
        ValueError: If model_name is not registered.
    """
    model_cls = get_model_class(model_name)

    if config is None:
        if model_cls is BaselineCNN:
            p = Path(config_path)
            if p.exists():
                b_cfg = BaselineCNNConfig.from_yaml(p)
            else:
                b_cfg = BaselineCNNConfig(**kwargs)
            return BaselineCNN(config=b_cfg, **kwargs)
        elif model_cls in (ResNet18Transfer, MobileNetV3SmallTransfer):
            t_cfg = TransferLearningConfig(model_name=model_name, **kwargs)
            return model_cls(config=t_cfg, **kwargs)
        return model_cls(**kwargs)

    if isinstance(config, (BaselineCNNConfig, TransferLearningConfig)):
        return model_cls(config=config, **kwargs)

    if isinstance(config, dict):
        if model_cls is BaselineCNN:
            b_cfg = BaselineCNNConfig.from_dict(config)
            return BaselineCNN(config=b_cfg, **kwargs)
        elif model_cls in (ResNet18Transfer, MobileNetV3SmallTransfer):
            t_cfg = TransferLearningConfig.from_dict(config)
            return model_cls(config=t_cfg, **kwargs)
        return model_cls(config=config, **kwargs)

    return model_cls(config=config, **kwargs)
