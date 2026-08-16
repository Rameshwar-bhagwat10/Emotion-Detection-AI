"""Unit tests for Phase 03 Training Augmentation."""

from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision import transforms

from ml.preprocessing.augmentation import AugmentationConfig, build_augmentation_pipeline


def test_augmentation_config_from_dict() -> None:
    """Verify AugmentationConfig parsing from configuration dictionary."""
    config_dict = {
        "augmentation": {
            "horizontal_flip": {"enabled": True, "probability": 0.75},
            "rotation": {"enabled": False, "degrees": 15},
            "affine": {"enabled": True, "translate": [0.1, 0.1], "scale": [0.9, 1.1]},
            "color_jitter": {"enabled": True, "brightness": 0.2, "contrast": 0.2},
        }
    }
    cfg = AugmentationConfig.from_dict(config_dict)

    assert cfg.horizontal_flip_enabled is True
    assert cfg.horizontal_flip_prob == 0.75
    assert cfg.rotation_enabled is False
    assert cfg.rotation_degrees == 15.0
    assert cfg.affine_enabled is True
    assert cfg.affine_translate == (0.1, 0.1)
    assert cfg.affine_scale == (0.9, 1.1)
    assert cfg.color_jitter_enabled is True
    assert cfg.brightness == 0.2
    assert cfg.contrast == 0.2


def test_augmentation_pipeline_composition() -> None:
    """Verify build_augmentation_pipeline constructs the expected list of transforms."""
    cfg = AugmentationConfig(
        horizontal_flip_enabled=True,
        rotation_enabled=True,
        affine_enabled=True,
        color_jitter_enabled=False,
    )
    pipeline = build_augmentation_pipeline(cfg)

    # Expected: RandomHorizontalFlip, RandomRotation, RandomAffine
    assert len(pipeline) == 3
    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in pipeline)
    assert any(isinstance(t, transforms.RandomRotation) for t in pipeline)
    assert any(isinstance(t, transforms.RandomAffine) for t in pipeline)


def test_augmentation_execution_on_image() -> None:
    """Verify augmentation executes on a PIL Image and preserves resolution."""
    raw_arr = np.random.randint(0, 256, size=(48, 48), dtype=np.uint8)
    pil_img = Image.fromarray(raw_arr)

    pipeline = build_augmentation_pipeline()
    transform_chain = transforms.Compose(pipeline)

    augmented_img = transform_chain(pil_img)

    assert isinstance(augmented_img, Image.Image)
    assert augmented_img.size == (48, 48)
