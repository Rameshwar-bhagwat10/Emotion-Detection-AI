"""Training data augmentation strategies for facial expression recognition.

Provides conservative, identity-preserving augmentation pipelines specifically tuned
for 48x48 grayscale facial expressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torchvision import transforms


@dataclass
class AugmentationConfig:
    """Hyperparameter configuration for facial expression augmentation."""

    horizontal_flip_enabled: bool = True
    horizontal_flip_prob: float = 0.5

    rotation_enabled: bool = True
    rotation_degrees: float = 12.0

    affine_enabled: bool = True
    affine_translate: tuple[float, float] = (0.05, 0.05)
    affine_scale: tuple[float, float] = (0.95, 1.05)

    color_jitter_enabled: bool = False
    brightness: float = 0.1
    contrast: float = 0.1

    extra_transforms: list[Any] = field(default_factory=list)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any] | None = None) -> AugmentationConfig:
        """Construct AugmentationConfig from a configuration dictionary."""
        if not config_dict:
            return cls()

        aug_dict = config_dict.get("augmentation", config_dict)

        hflip = aug_dict.get("horizontal_flip", {})
        rot = aug_dict.get("rotation", {})
        aff = aug_dict.get("affine", {})
        cjitter = aug_dict.get("color_jitter", {})

        translate_val = aff.get("translate", [0.05, 0.05])
        if isinstance(translate_val, list):
            translate_tuple = (float(translate_val[0]), float(translate_val[1]))
        elif isinstance(translate_val, (int, float)):
            translate_tuple = (float(translate_val), float(translate_val))
        else:
            translate_tuple = (0.05, 0.05)

        scale_val = aff.get("scale", [0.95, 1.05])
        if isinstance(scale_val, list):
            scale_tuple = (float(scale_val[0]), float(scale_val[1]))
        elif isinstance(scale_val, (int, float)):
            scale_tuple = (float(scale_val), float(scale_val))
        else:
            scale_tuple = (0.95, 1.05)

        return cls(
            horizontal_flip_enabled=bool(hflip.get("enabled", True)),
            horizontal_flip_prob=float(hflip.get("probability", 0.5)),
            rotation_enabled=bool(rot.get("enabled", True)),
            rotation_degrees=float(rot.get("degrees", 12.0)),
            affine_enabled=bool(aff.get("enabled", True)),
            affine_translate=translate_tuple,
            affine_scale=scale_tuple,
            color_jitter_enabled=bool(cjitter.get("enabled", False)),
            brightness=float(cjitter.get("brightness", 0.1)),
            contrast=float(cjitter.get("contrast", 0.1)),
        )


def build_augmentation_pipeline(
    config: AugmentationConfig | dict[str, Any] | None = None,
) -> list[Any]:
    """Construct a list of training-only augmentation transforms.

    Args:
        config: AugmentationConfig instance or dictionary.

    Returns:
        list of torchvision transforms to be applied on PIL Images or Tensors.
    """
    if config is None or isinstance(config, dict):
        cfg = AugmentationConfig.from_dict(config)
    else:
        cfg = config

    pipeline: list[Any] = []

    # 1. Random Horizontal Flip
    if cfg.horizontal_flip_enabled and cfg.horizontal_flip_prob > 0:
        pipeline.append(transforms.RandomHorizontalFlip(p=cfg.horizontal_flip_prob))

    # 2. Random Rotation (subtle, preserves facial orientation)
    if cfg.rotation_enabled and cfg.rotation_degrees > 0:
        pipeline.append(
            transforms.RandomRotation(
                degrees=(-cfg.rotation_degrees, cfg.rotation_degrees),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )

    # 3. Random Affine (translation & scale)
    if cfg.affine_enabled:
        pipeline.append(
            transforms.RandomAffine(
                degrees=0,
                translate=cfg.affine_translate,
                scale=cfg.affine_scale,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )

    # 4. Color Jitter (optional brightness / contrast)
    if cfg.color_jitter_enabled:
        pipeline.append(
            transforms.ColorJitter(
                brightness=cfg.brightness,
                contrast=cfg.contrast,
            )
        )

    if cfg.extra_transforms:
        pipeline.extend(cfg.extra_transforms)

    return pipeline
