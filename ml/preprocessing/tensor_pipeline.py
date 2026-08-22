"""Ultra-high-speed pure PyTorch tensor batching and transformation engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Grayscale & ImageNet normalization parameters
GRAY_MEAN = 0.507743
GRAY_STD = 0.255009
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class TensorDataStore:
    """Stores full split tensors in RAM for microsecond batch slicing."""

    def __init__(self, split: str = "train", processed_dir: str | Path | None = None) -> None:
        p_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
        npz_file = p_dir / f"fer2013_{split}.npz"
        if not npz_file.exists():
            raise FileNotFoundError(f"Missing {npz_file}")

        data = np.load(npz_file, allow_pickle=True)
        images_np = data["images"]  # (N, 48, 48) uint8
        labels_np = data["labels"]  # (N,) int64

        # Convert to float32 tensor [N, 1, 48, 48] in range [0.0, 1.0]
        self.images = torch.from_numpy(images_np).unsqueeze(1).float() / 255.0
        self.labels = torch.from_numpy(labels_np).long()
        self.num_samples = len(self.labels)
        self.split = split

    def get_class_counts(self) -> list[int]:
        counts = torch.bincount(self.labels, minlength=7)
        return counts.tolist()

    def get_sample_weights(self, smoothing: float = 0.5) -> torch.Tensor:
        counts = torch.bincount(self.labels, minlength=7).float()
        weights = 1.0 / (torch.pow(counts, smoothing) + 1e-6)
        weights = weights / weights.mean()
        return weights[self.labels]


def apply_tensor_batch_augment(
    batch: torch.Tensor,
    aug_level: str = "moderate",
) -> torch.Tensor:
    """Apply vectorized random augmentations directly on batched tensor [B, C, H, W]."""
    b, c, h, w = batch.shape

    # 1. Random Horizontal Flip (50% probability per sample)
    flip_mask = torch.rand(b) > 0.5
    if flip_mask.any():
        batch[flip_mask] = torch.flip(batch[flip_mask], dims=[-1])

    if aug_level in ("conservative", "moderate", "erasing"):
        # 2. Random slight translation (+-3 pixels)
        shift_x = torch.randint(-3, 4, (1,)).item()
        shift_y = torch.randint(-3, 4, (1,)).item()
        if shift_x != 0 or shift_y != 0:
            batch = torch.roll(batch, shifts=(shift_y, shift_x), dims=(-2, -1))

    if aug_level in ("moderate", "erasing"):
        # 3. Random Contrast & Brightness Jitter
        scale = (torch.rand(b, 1, 1, 1) * 0.3 + 0.85)  # [0.85, 1.15]
        shift = (torch.rand(b, 1, 1, 1) * 0.2 - 0.1)   # [-0.10, +0.10]
        batch = torch.clamp(batch * scale + shift, 0.0, 1.0)

    if aug_level == "erasing":
        # 4. Random Cutout / Erasing
        erase_mask = torch.rand(b) > 0.7
        if erase_mask.any():
            eh, ew = int(h * 0.2), int(w * 0.2)
            ey = torch.randint(0, h - eh, (1,)).item()
            ex = torch.randint(0, w - ew, (1,)).item()
            batch[erase_mask, :, ey : ey + eh, ex : ex + ew] = 0.5

    return batch


class FastTensorDataLoader:
    """High-throughput tensor batch generator executing in C++ speed."""

    def __init__(
        self,
        store: TensorDataStore,
        batch_size: int = 128,
        shuffle: bool = True,
        sample_weights: torch.Tensor | None = None,
        input_size: tuple[int, int] = (48, 48),
        channels: int = 1,
        augment: bool = False,
        aug_level: str = "moderate",
        limit_samples: int | None = None,
    ) -> None:
        self.store = store
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.input_size = input_size
        self.channels = channels
        self.augment = augment
        self.aug_level = aug_level

        if limit_samples and limit_samples < store.num_samples:
            self.total_samples = limit_samples
        else:
            self.total_samples = store.num_samples

    def __len__(self) -> int:
        return (self.total_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if self.sample_weights is not None:
            # Weighted random sampling with replacement
            indices = torch.multinomial(
                self.sample_weights, num_samples=self.total_samples, replacement=True
            )
        elif self.shuffle:
            indices = torch.randperm(self.store.num_samples)[: self.total_samples]
        else:
            indices = torch.arange(self.total_samples)

        for i in range(0, self.total_samples, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            batch_imgs = self.store.images[batch_idx].clone()
            batch_labels = self.store.labels[batch_idx]

            # Augmentation
            if self.augment:
                batch_imgs = apply_tensor_batch_augment(batch_imgs, aug_level=self.aug_level)

            # Spatial resize if needed
            if self.input_size != (48, 48):
                batch_imgs = F.interpolate(
                    batch_imgs, size=self.input_size, mode="bilinear", align_corners=False
                )

            # Channels
            if self.channels == 3:
                batch_imgs = batch_imgs.repeat(1, 3, 1, 1)
                batch_imgs = (batch_imgs - IMAGENET_MEAN) / IMAGENET_STD
            else:
                batch_imgs = (batch_imgs - GRAY_MEAN) / GRAY_STD

            yield batch_imgs, batch_labels
