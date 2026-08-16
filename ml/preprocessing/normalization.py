"""Dataset normalization utilities and training statistics calculator.

Provides exact TRAIN-split normalization statistics calculation, tensor normalization,
and inverse normalization for visualization.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from ml.datasets.fer2013.loader import FER2013Loader
from ml.datasets.fer2013.parser import DatasetRecord

DEFAULT_TRAIN_MEAN = 0.507743
DEFAULT_TRAIN_STD = 0.255009


@dataclass(frozen=True)
class NormalizationStats:
    """Statistics calculated strictly from the training split."""

    mean: float
    std: float
    raw_mean: float
    raw_std: float
    total_samples: int
    split: str

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary."""
        return asdict(self)


def calculate_train_normalization_stats(
    source: FER2013Loader | Sequence[DatasetRecord],
    split: str = "train",
) -> NormalizationStats:
    """Calculate exact pixel mean and standard deviation from the training split.

    Args:
        source: FER2013Loader instance or sequence of DatasetRecords.
        split: The split name to filter on (default: 'train').

    Returns:
        NormalizationStats: Computed statistics for normalized [0, 1] and raw [0, 255] ranges.

    Raises:
        ValueError: If no records are found in the specified split.
    """
    if isinstance(source, FER2013Loader):
        records = source.load_records(split=split)
    else:
        records = [r for r in source if r.split == split]

    if not records:
        raise ValueError(
            f"No records found for split '{split}' to compute normalization statistics."
        )

    # Convert all records to float32 scaled to [0, 1]
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0

    raw_pixel_sum = 0.0
    raw_pixel_sq_sum = 0.0

    for record in records:
        img_arr = record.image.astype(np.float64)
        scaled_arr = img_arr / 255.0

        pixel_sum += float(scaled_arr.sum())
        pixel_sq_sum += float((scaled_arr**2).sum())
        total_pixels += scaled_arr.size

        raw_pixel_sum += float(img_arr.sum())
        raw_pixel_sq_sum += float((img_arr**2).sum())

    # [0, 1] metrics
    mean_val = float(pixel_sum / total_pixels)
    var_val = float((pixel_sq_sum / total_pixels) - (mean_val**2))
    std_val = float(np.sqrt(max(0.0, var_val)))

    # [0, 255] metrics
    raw_mean_val = float(raw_pixel_sum / total_pixels)
    raw_var_val = float((raw_pixel_sq_sum / total_pixels) - (raw_mean_val**2))
    raw_std_val = float(np.sqrt(max(0.0, raw_var_val)))

    return NormalizationStats(
        mean=round(mean_val, 6),
        std=round(std_val, 6),
        raw_mean=round(raw_mean_val, 6),
        raw_std=round(raw_std_val, 6),
        total_samples=len(records),
        split=split,
    )


def normalize(
    tensor: torch.Tensor,
    mean: float | Sequence[float] = DEFAULT_TRAIN_MEAN,
    std: float | Sequence[float] = DEFAULT_TRAIN_STD,
) -> torch.Tensor:
    """Normalize a PyTorch image tensor using (x - mean) / std.

    Args:
        tensor: Float tensor of shape [C, H, W] or [B, C, H, W] in range [0, 1].
        mean: Channel mean value(s).
        std: Channel standard deviation value(s).

    Returns:
        torch.Tensor: Normalized tensor.

    Raises:
        ValueError: If std is zero or negative.
    """
    if isinstance(mean, (int, float)):
        mean_t = torch.tensor([mean], dtype=tensor.dtype, device=tensor.device)
    else:
        mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)

    if isinstance(std, (int, float)):
        if std <= 0:
            raise ValueError(f"Standard deviation must be strictly positive, got: {std}")
        std_t = torch.tensor([std], dtype=tensor.dtype, device=tensor.device)
    else:
        for s in std:
            if s <= 0:
                raise ValueError(f"Standard deviation values must be strictly positive, got: {std}")
        std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    # Reshape for broadcasting
    if tensor.ndim == 3:
        # [C, H, W]
        mean_t = mean_t.view(-1, 1, 1)
        std_t = std_t.view(-1, 1, 1)
    elif tensor.ndim == 4:
        # [B, C, H, W]
        mean_t = mean_t.view(1, -1, 1, 1)
        std_t = std_t.view(1, -1, 1, 1)
    elif tensor.ndim == 2:
        # [H, W]
        mean_t = mean_t.view(1, 1)
        std_t = std_t.view(1, 1)

    return (tensor - mean_t) / std_t


def denormalize(
    tensor: torch.Tensor,
    mean: float | Sequence[float] = DEFAULT_TRAIN_MEAN,
    std: float | Sequence[float] = DEFAULT_TRAIN_STD,
    to_255: bool = False,
) -> torch.Tensor:
    """Invert normalization: (x * std) + mean and clamp to valid range.

    Args:
        tensor: Normalized float tensor of shape [C, H, W] or [B, C, H, W].
        mean: Channel mean value(s).
        std: Channel standard deviation value(s).
        to_255: If True, scale to [0, 255] range; otherwise scale to [0, 1].

    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    if isinstance(mean, (int, float)):
        mean_t = torch.tensor([mean], dtype=tensor.dtype, device=tensor.device)
    else:
        mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)

    if isinstance(std, (int, float)):
        std_t = torch.tensor([std], dtype=tensor.dtype, device=tensor.device)
    else:
        std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    if tensor.ndim == 3:
        mean_t = mean_t.view(-1, 1, 1)
        std_t = std_t.view(-1, 1, 1)
    elif tensor.ndim == 4:
        mean_t = mean_t.view(1, -1, 1, 1)
        std_t = std_t.view(1, -1, 1, 1)
    elif tensor.ndim == 2:
        mean_t = mean_t.view(1, 1)
        std_t = std_t.view(1, 1)

    denorm = (tensor * std_t) + mean_t
    denorm = torch.clamp(denorm, 0.0, 1.0)

    if to_255:
        denorm = torch.clamp(denorm * 255.0, 0.0, 255.0)

    return denorm
