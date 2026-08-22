"""High-speed in-memory dataset loader from pre-cached NumPy archives."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# FER2013 training split statistics (grayscale)
DEFAULT_GRAY_MEAN = 0.507743
DEFAULT_GRAY_STD = 0.255009

# ImageNet statistics for RGB models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_v2_transforms(
    split: str = "train",
    input_size: tuple[int, int] = (48, 48),
    channels: int = 1,
    augment: bool = True,
    aug_strength: str = "moderate",
) -> transforms.Compose:
    """Build torchvision transform pipeline tailored for Model V2 experiments."""
    t_list: list[Any] = [transforms.ToPILImage()]

    # Resize if target size differs from 48x48
    if input_size != (48, 48):
        t_list.append(transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR))

    if split == "train" and augment:
        if aug_strength == "conservative":
            t_list.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
                ]
            )
        elif aug_strength == "moderate":
            t_list.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=12),
                    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                ]
            )
        elif aug_strength == "erasing":
            t_list.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=12),
                    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                ]
            )

    t_list.append(transforms.ToTensor())

    # Channels & Normalization
    if channels == 3:
        # Convert 1-channel tensor [1, H, W] to 3-channel [3, H, W]
        t_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))
        t_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        t_list.append(transforms.Normalize(mean=[DEFAULT_GRAY_MEAN], std=[DEFAULT_GRAY_STD]))

    if split == "train" and augment and aug_strength == "erasing":
        t_list.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), value="random"))

    return transforms.Compose(t_list)


class FastFER2013Dataset(Dataset[dict[str, Any]]):
    """Memory-resident dataset loading from .npz archives in milliseconds."""

    def __init__(
        self,
        split: str = "train",
        processed_dir: str | Path | None = None,
        transform: Callable[[Any], torch.Tensor] | None = None,
    ) -> None:
        self.split = split.lower().strip()
        p_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
        npz_file = p_dir / f"fer2013_{self.split}.npz"

        if not npz_file.exists():
            raise FileNotFoundError(
                f"Cached dataset not found at {npz_file}. Run scripts/data/cache_processed_arrays.py first."
            )

        data = np.load(npz_file, allow_pickle=True)
        self.images: np.ndarray = data["images"]  # Shape (N, 48, 48) uint8
        self.labels: np.ndarray = data["labels"]  # Shape (N,) int64
        self.record_ids: np.ndarray = data["record_ids"]
        self.emotions: np.ndarray = data["emotions"]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        img_np = self.images[index]  # (48, 48) uint8
        label = int(self.labels[index])

        if self.transform is not None:
            img_tensor = self.transform(img_np)
        else:
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).float() / 255.0

        return {
            "image": img_tensor,
            "label": label,
            "record_id": int(self.record_ids[index]),
            "emotion": str(self.emotions[index]),
            "split": self.split,
        }

    def get_class_counts(self) -> dict[int, int]:
        counts = {}
        for c in range(7):
            counts[c] = int((self.labels == c).sum())
        return counts

    def get_sample_weights(self, smoothing: float = 0.5) -> torch.Tensor:
        """Compute per-sample weights for WeightedRandomSampler with optional smoothing."""
        class_counts = np.bincount(self.labels, minlength=7).astype(np.float32)
        # Power smoothing: power=1.0 is full inverse frequency, power=0.5 is sqrt inverse frequency
        class_weights = 1.0 / (np.power(class_counts, smoothing) + 1e-6)
        class_weights = class_weights / np.mean(class_weights)
        sample_weights = class_weights[self.labels]
        return torch.from_numpy(sample_weights).float()
