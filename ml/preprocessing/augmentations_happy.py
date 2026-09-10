"""Targeted Happy Emotion Facial Data Augmentations."""

from __future__ import annotations

import math
import random
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF


class TargetedHappyAugmenter(nn.Module):
    """Targeted augmentation pipeline specifically designed to preserve Duchenne facial markers
    (subtle smile, mouth corners, zygomatic major muscle activity, eye crinkles)
    while introducing realistic geometric and photometric variation."""

    def __init__(
        self,
        p_flip: float = 0.5,
        max_rotation_deg: float = 8.0,
        max_translation_pct: float = 0.04,
        brightness_delta: float = 0.12,
        contrast_delta: float = 0.12,
    ) -> None:
        super().__init__()
        self.p_flip = p_flip
        self.max_rotation_deg = max_rotation_deg
        self.max_translation_pct = max_translation_pct
        self.brightness_delta = brightness_delta
        self.contrast_delta = contrast_delta

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply targeted facial augmentation on image tensor [C, H, W] or batch [B, C, H, W].

        Expects normalized or float tensors in range [0, 1] or ImageNet normalized.
        """
        # Horizontal Flip (preserves symmetric facial geometry)
        if random.random() < self.p_flip:
            img = TF.horizontal_flip(img)

        # Subtle Affine Transform (small rotation +-8 deg, translation +-4%)
        angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
        if isinstance(img, torch.Tensor) and img.ndim == 4:
            _, _, h, w = img.shape
        else:
            _, h, w = img.shape

        max_dx = int(w * self.max_translation_pct)
        max_dy = int(h * self.max_translation_pct)
        translations = (random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy))

        # Scale factor (0.97 - 1.03)
        scale = random.uniform(0.97, 1.03)

        img = TF.affine(
            img,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=T.InterpolationMode.BILINEAR,
        )

        # Controlled Photometric Jitter (brightness & contrast)
        brightness_factor = 1.0 + random.uniform(-self.brightness_delta, self.brightness_delta)
        contrast_factor = 1.0 + random.uniform(-self.contrast_delta, self.contrast_delta)

        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)

        return img


def get_happy_augmentation_transform() -> TargetedHappyAugmenter:
    """Return configured instance of TargetedHappyAugmenter."""
    return TargetedHappyAugmenter()
