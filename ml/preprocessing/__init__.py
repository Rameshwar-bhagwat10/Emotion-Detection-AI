"""FER2013 Preprocessing & Data Augmentation Package.

Provides PyTorch Datasets, transforms, DataLoaders, and train-only normalization.
"""

from ml.preprocessing.augmentation import AugmentationConfig, build_augmentation_pipeline
from ml.preprocessing.dataloaders import (
    build_dataloaders,
    create_test_loader,
    create_train_loader,
    create_val_loader,
)
from ml.preprocessing.datasets import FER2013Dataset
from ml.preprocessing.normalization import (
    DEFAULT_TRAIN_MEAN,
    DEFAULT_TRAIN_STD,
    NormalizationStats,
    calculate_train_normalization_stats,
    denormalize,
    normalize,
)
from ml.preprocessing.transforms import (
    build_test_transform,
    build_train_transform,
    build_val_transform,
    load_preprocessing_config,
)

__all__ = [
    "AugmentationConfig",
    "build_augmentation_pipeline",
    "build_dataloaders",
    "create_train_loader",
    "create_val_loader",
    "create_test_loader",
    "FER2013Dataset",
    "DEFAULT_TRAIN_MEAN",
    "DEFAULT_TRAIN_STD",
    "NormalizationStats",
    "calculate_train_normalization_stats",
    "normalize",
    "denormalize",
    "build_train_transform",
    "build_val_transform",
    "build_test_transform",
    "load_preprocessing_config",
]
