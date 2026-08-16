"""Transform composition for FER2013 training, validation, and evaluation pipelines.

Provides deterministic model-ready PyTorch tensor generation with strict split isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from torchvision import transforms

from ml.preprocessing.augmentation import AugmentationConfig, build_augmentation_pipeline
from ml.preprocessing.normalization import DEFAULT_TRAIN_MEAN, DEFAULT_TRAIN_STD

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "preprocessing.yaml"


def load_preprocessing_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load preprocessing configuration from YAML file."""
    target_path = Path(config_path) if config_path else CONFIG_PATH
    if not target_path.exists():
        return {}
    with open(target_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return dict(data) if isinstance(data, dict) else {}


def extract_normalization_params(
    config: dict[str, Any] | None = None,
) -> tuple[list[float], list[float]]:
    """Extract mean and standard deviation from config dictionary or defaults."""
    if not config:
        config = load_preprocessing_config()

    norm_dict = config.get("normalization", {})
    mean_val = norm_dict.get("mean", [DEFAULT_TRAIN_MEAN])
    std_val = norm_dict.get("std", [DEFAULT_TRAIN_STD])

    if isinstance(mean_val, (int, float)):
        mean_list = [float(mean_val)]
    else:
        mean_list = [float(m) for m in mean_val]

    if isinstance(std_val, (int, float)):
        std_list = [float(std_val)]
    else:
        std_list = [float(s) for s in std_val]

    return mean_list, std_list


def build_train_transform(
    config: dict[str, Any] | None = None,
    mean: float | list[float] | None = None,
    std: float | list[float] | None = None,
    augmentation_config: AugmentationConfig | None = None,
) -> transforms.Compose:
    """Build the model-ready transform pipeline for the TRAINING split.

    Pipeline:
        1. Convert ndarray to PIL Image
        2. Apply train-only random augmentations (flip, rotation, affine)
        3. Convert PIL Image to FloatTensor [1, 48, 48] and scale [0, 255] -> [0.0, 1.0]
        4. Normalize using TRAIN mean and standard deviation

    Args:
        config: Optional full preprocessing config dictionary.
        mean: Optional explicit mean value(s).
        std: Optional explicit std value(s).
        augmentation_config: Optional explicit AugmentationConfig.

    Returns:
        torchvision.transforms.Compose: Training transform pipeline.
    """
    if config is None:
        config = load_preprocessing_config()

    if mean is None or std is None:
        cfg_mean, cfg_std = extract_normalization_params(config)
        mean_list = (
            [mean] if isinstance(mean, (int, float)) else (mean if mean is not None else cfg_mean)
        )
        std_list = [std] if isinstance(std, (int, float)) else (std if std is not None else cfg_std)
    else:
        mean_list = [mean] if isinstance(mean, (int, float)) else list(mean)
        std_list = [std] if isinstance(std, (int, float)) else list(std)

    if augmentation_config is None:
        aug_pipeline = build_augmentation_pipeline(config)
    else:
        aug_pipeline = build_augmentation_pipeline(augmentation_config)

    transform_list: list[Any] = [
        transforms.ToPILImage(),
        *aug_pipeline,
        transforms.ToTensor(),  # Converts uint8 [0, 255] to float32 [0.0, 1.0] with shape [1, 48, 48]
        transforms.Normalize(mean=mean_list, std=std_list),
    ]

    return transforms.Compose(transform_list)


def build_val_transform(
    config: dict[str, Any] | None = None,
    mean: float | list[float] | None = None,
    std: float | list[float] | None = None,
) -> transforms.Compose:
    """Build the model-ready transform pipeline for the VALIDATION split.

    Pipeline:
        1. Convert ndarray to PIL Image
        2. Convert to FloatTensor [1, 48, 48] and scale [0, 255] -> [0.0, 1.0]
        3. Normalize using TRAIN mean and standard deviation
        (NO random augmentations - 100% deterministic)

    Args:
        config: Optional preprocessing config dictionary.
        mean: Optional explicit mean value(s).
        std: Optional explicit std value(s).

    Returns:
        torchvision.transforms.Compose: Validation transform pipeline.
    """
    if config is None:
        config = load_preprocessing_config()

    if mean is None or std is None:
        cfg_mean, cfg_std = extract_normalization_params(config)
        mean_list = (
            [mean] if isinstance(mean, (int, float)) else (mean if mean is not None else cfg_mean)
        )
        std_list = [std] if isinstance(std, (int, float)) else (std if std is not None else cfg_std)
    else:
        mean_list = [mean] if isinstance(mean, (int, float)) else list(mean)
        std_list = [std] if isinstance(std, (int, float)) else list(std)

    transform_list: list[Any] = [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_list, std=std_list),
    ]

    return transforms.Compose(transform_list)


def build_test_transform(
    config: dict[str, Any] | None = None,
    mean: float | list[float] | None = None,
    std: float | list[float] | None = None,
) -> transforms.Compose:
    """Build the model-ready transform pipeline for the TEST split.

    Pipeline:
        1. Convert ndarray to PIL Image
        2. Convert to FloatTensor [1, 48, 48] and scale [0, 255] -> [0.0, 1.0]
        3. Normalize using TRAIN mean and standard deviation
        (NO random augmentations - 100% deterministic)

    Args:
        config: Optional preprocessing config dictionary.
        mean: Optional explicit mean value(s).
        std: Optional explicit std value(s).

    Returns:
        torchvision.transforms.Compose: Test transform pipeline.
    """
    return build_val_transform(config=config, mean=mean, std=std)
