"""DataLoader creation utilities for FER2013 training and evaluation pipelines.

Provides model-ready batched data streaming with configurable worker, batch size,
and shuffling configurations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ml.preprocessing.datasets import FER2013Dataset
from ml.preprocessing.transforms import load_preprocessing_config


def create_train_loader(
    dataset: Dataset[dict[str, Any]],
    batch_size: int = 64,
    shuffle: bool = True,
    sampler: torch.utils.data.Sampler[Any] | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    seed: int | None = None,
) -> DataLoader[dict[str, Any]]:
    """Create a PyTorch DataLoader for the TRAINING split.

    Args:
        dataset: Training dataset instance.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle samples each epoch (default: True, ignored if sampler is provided).
        sampler: Optional custom Sampler (e.g. WeightedRandomSampler).
        num_workers: Subprocess workers for data loading (default: 0 for cross-platform stability).
        pin_memory: If True, pins memory for faster GPU transfer.
        drop_last: If True, drops the last incomplete batch.
        seed: Optional random seed for deterministic DataLoader generator.

    Returns:
        DataLoader: Configured training DataLoader.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    # Pin memory only if CUDA is available or explicitly requested on GPU systems
    pin_mem = pin_memory and torch.cuda.is_available()
    do_shuffle = shuffle if sampler is None else False

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=do_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        generator=generator,
    )


def create_val_loader(
    dataset: Dataset[dict[str, Any]],
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader[dict[str, Any]]:
    """Create a PyTorch DataLoader for the VALIDATION split.

    Args:
        dataset: Validation dataset instance.
        batch_size: Mini-batch size.
        num_workers: Subprocess workers for data loading.
        pin_memory: If True, pins memory for faster GPU transfer.
        drop_last: If True, drops the last incomplete batch.

    Returns:
        DataLoader: Configured validation DataLoader (shuffle=False).
    """
    pin_mem = pin_memory and torch.cuda.is_available()

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # Strict non-shuffling for deterministic evaluation
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )


def create_test_loader(
    dataset: Dataset[dict[str, Any]],
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader[dict[str, Any]]:
    """Create a PyTorch DataLoader for the TEST split.

    Args:
        dataset: Test dataset instance.
        batch_size: Mini-batch size.
        num_workers: Subprocess workers for data loading.
        pin_memory: If True, pins memory for faster GPU transfer.
        drop_last: If True, drops the last incomplete batch.

    Returns:
        DataLoader: Configured test DataLoader (shuffle=False).
    """
    return create_val_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def build_dataloaders(
    data_path: str | Path = "data/raw/fer2013/fer2013.csv",
    config_path: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    """Build train, validation, and test DataLoaders from configuration.

    Args:
        data_path: Path to raw dataset CSV or directory.
        config_path: Optional path to preprocessing.yaml.
        batch_size: Optional override for batch size.
        num_workers: Optional override for num_workers.

    Returns:
        tuple of (train_loader, val_loader, test_loader).
    """
    config = load_preprocessing_config(config_path)
    dl_config = config.get("dataloader", {})
    rep_config = config.get("reproducibility", {})

    b_size = batch_size if batch_size is not None else int(dl_config.get("batch_size", 64))
    n_workers = num_workers if num_workers is not None else int(dl_config.get("num_workers", 0))
    pin_mem = bool(dl_config.get("pin_memory", True))
    drop_train = bool(dl_config.get("drop_last_train", False))
    drop_eval = bool(dl_config.get("drop_last_eval", False))
    seed = int(rep_config.get("seed", 42))

    # Instantiate datasets
    train_dataset = FER2013Dataset(split="train", data_path=data_path, config=config)
    val_dataset = FER2013Dataset(split="val", data_path=data_path, config=config)
    test_dataset = FER2013Dataset(split="test", data_path=data_path, config=config)

    # Build DataLoaders
    train_loader = create_train_loader(
        dataset=train_dataset,
        batch_size=b_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=pin_mem,
        drop_last=drop_train,
        seed=seed,
    )

    val_loader = create_val_loader(
        dataset=val_dataset,
        batch_size=b_size,
        num_workers=n_workers,
        pin_memory=pin_mem,
        drop_last=drop_eval,
    )

    test_loader = create_test_loader(
        dataset=test_dataset,
        batch_size=b_size,
        num_workers=n_workers,
        pin_memory=pin_mem,
        drop_last=drop_eval,
    )

    return train_loader, val_loader, test_loader
