"""Sanity check and validation runner for Phase 03 Preprocessing & DataLoaders.

Verifies dataset cardinalities, tensor dimensions, data types, label consistency,
DataLoader batch shapes, and determinism.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402

from ml.preprocessing.dataloaders import build_dataloaders  # noqa: E402
from ml.preprocessing.datasets import FER2013Dataset  # noqa: E402
from ml.preprocessing.normalization import (  # noqa: E402
    DEFAULT_TRAIN_MEAN,
    DEFAULT_TRAIN_STD,
    denormalize,
)
from ml.preprocessing.transforms import build_val_transform  # noqa: E402
from ml.utils.seed import set_seed  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("verify-preprocessing")


def run_preprocessing_verification(data_path: str | Path = "data/raw/fer2013/fer2013.csv") -> bool:
    """Execute complete Phase 03 sanity check."""
    set_seed(42)
    csv_file = ROOT_DIR / data_path
    if not csv_file.exists():
        logger.error("Dataset not found at: %s", csv_file)
        return False

    logger.info("Instantiating FER2013 Datasets...")
    train_ds = FER2013Dataset(split="train", data_path=csv_file)
    val_ds = FER2013Dataset(split="val", data_path=csv_file)
    test_ds = FER2013Dataset(split="test", data_path=csv_file)

    # 1. Check cardinalities
    len_train = len(train_ds)
    len_val = len(val_ds)
    len_test = len(test_ds)

    assert len_train == 28709, f"Train count mismatch: expected 28709, got {len_train}"
    assert len_val == 3589, f"Val count mismatch: expected 3589, got {len_val}"
    assert len_test == 3589, f"Test count mismatch: expected 3589, got {len_test}"

    # 2. Check sample contracts
    sample_train = train_ds[0]
    sample_val = val_ds[0]
    sample_test = test_ds[0]

    for name, sample in [("Train", sample_train), ("Val", sample_val), ("Test", sample_test)]:
        img = sample["image"]
        lbl = sample["label"]
        emo = sample["emotion"]

        assert isinstance(img, torch.Tensor), f"{name} image is not a torch.Tensor"
        assert img.shape == (
            1,
            48,
            48,
        ), f"{name} shape mismatch: expected (1, 48, 48), got {img.shape}"
        assert (
            img.dtype == torch.float32
        ), f"{name} dtype mismatch: expected float32, got {img.dtype}"
        assert isinstance(lbl, int) and 0 <= lbl <= 6, f"{name} invalid label: {lbl}"
        assert isinstance(emo, str) and len(emo) > 0, f"{name} invalid emotion string: {emo}"

    # 3. Check DataLoaders
    logger.info("Building DataLoaders...")
    train_loader, val_loader, test_loader = build_dataloaders(data_path=csv_file, batch_size=64)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    assert train_batch["image"].shape == (
        64,
        1,
        48,
        48,
    ), f"Train batch shape: {train_batch['image'].shape}"
    assert train_batch["image"].dtype == torch.float32
    assert train_batch["label"].shape == (64,)
    assert val_batch["image"].shape == (64, 1, 48, 48)
    assert test_batch["image"].shape == (64, 1, 48, 48)

    # 4. Check determinism of validation transforms
    val_t = build_val_transform()
    raw_img = val_ds.records[0].image
    t1 = val_t(raw_img)
    t2 = val_t(raw_img)
    assert torch.allclose(t1, t2), "Validation transform must be 100% deterministic"

    # 5. Check denormalization
    denorm = denormalize(sample_train["image"], mean=DEFAULT_TRAIN_MEAN, std=DEFAULT_TRAIN_STD)
    assert (
        denorm.min() >= 0.0 and denorm.max() <= 1.0
    ), f"Denormalized bounds violated: [{denorm.min()}, {denorm.max()}]"

    # 6. Check class weights
    class_weights = train_ds.get_class_weights()
    assert class_weights.shape == (7,)
    assert (class_weights > 0).all()

    print("\n" + "=" * 54)
    print("PHASE 03 PREPROCESSING & DATALOADER VERIFICATION")
    print("=" * 54)
    print("Dataset Cardinality:")
    print(f"  Train:      {len_train:,} (Verified)")
    print(f"  Validation: {len_val:,} (Verified)")
    print(f"  Test:       {len_test:,} (Verified)")
    print("-" * 54)
    print("Tensor Contract:")
    print(f"  Image Shape: {sample_train['image'].shape} (Channels=1, H=48, W=48)")
    print(f"  Image Dtype: {sample_train['image'].dtype}")
    print("  Label Range: [0..6] (7 Emotion Classes)")
    print("-" * 54)
    print("DataLoader Batches:")
    print(f"  Train Batch: {train_batch['image'].shape} (shuffle=True)")
    print(f"  Val Batch:   {val_batch['image'].shape} (shuffle=False)")
    print(f"  Test Batch:  {test_batch['image'].shape} (shuffle=False)")
    print("-" * 54)
    print("Determinism & Normalization:")
    print("  Val Transform Deterministic: YES (100% Match)")
    print(
        f"  Train-only Mean/Std Applied: mu={DEFAULT_TRAIN_MEAN:.6f}, sigma={DEFAULT_TRAIN_STD:.6f}"
    )
    print(f"  Denormalization Bounds:      [{denorm.min():.2f}, {denorm.max():.2f}]")
    print("-" * 54)
    print("STATUS: PASS (100% Verified)")
    print("=" * 54 + "\n")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify FER2013 Preprocessing & DataLoaders")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/fer2013/fer2013.csv",
        help="Path to raw FER2013 CSV",
    )
    args = parser.parse_args()

    success = run_preprocessing_verification(args.data_path)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
