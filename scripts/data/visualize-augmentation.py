"""Generate visual comparison and sample grids for Phase 03 data augmentations.

Demonstrates that conservative facial augmentations preserve identity, facial orientation,
and emotion morphology without extreme distortion.
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from torchvision import transforms  # noqa: E402

from ml.datasets.fer2013.loader import FER2013Loader  # noqa: E402
from ml.datasets.fer2013.parser import DatasetRecord  # noqa: E402
from ml.preprocessing.normalization import (  # noqa: E402
    DEFAULT_TRAIN_MEAN,
    DEFAULT_TRAIN_STD,
    denormalize,
)
from ml.preprocessing.transforms import build_train_transform  # noqa: E402
from ml.utils.seed import set_seed  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("visualize-augmentation")


def generate_preprocessing_comparison(
    records: list[DatasetRecord],
    output_path: Path,
) -> None:
    """Generate side-by-side comparison grid across all 7 emotions."""
    # Find 1 sample per emotion
    emotion_order = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    chosen_records: dict[str, DatasetRecord] = {}

    for rec in records:
        if rec.label_name in emotion_order and rec.label_name not in chosen_records:
            chosen_records[rec.label_name] = rec
        if len(chosen_records) == len(emotion_order):
            break

    fig, axes = plt.subplots(7, 5, figsize=(14, 18), dpi=300)
    fig.suptitle(
        "FER2013 — Data Augmentation & Preprocessing Pipeline Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    column_titles = [
        "1. Raw Image\n(48x48 uint8)",
        "2. Horizontal Flip\n(p=1.0)",
        "3. Rotation\n(12 deg)",
        "4. Affine\n(Trans + Scale)",
        "5. Final Composite\n(Denormalized)",
    ]

    hflip_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=1.0)]
    )
    rot_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomRotation(degrees=(12, 12))]
    )
    affine_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(1.05, 1.05)),
        ]
    )
    composite_transform = build_train_transform()

    for row_idx, emotion in enumerate(emotion_order):
        rec = chosen_records.get(emotion)
        if rec is None:
            continue

        raw_img = rec.image  # (48, 48)

        # 1. Raw
        axes[row_idx, 0].imshow(raw_img, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 0].set_ylabel(
            f"{emotion.upper()}\n(Label {rec.label})",
            fontsize=11,
            fontweight="bold",
            rotation=0,
            labelpad=50,
            va="center",
        )

        # 2. HFlip
        axes[row_idx, 1].imshow(hflip_transform(raw_img), cmap="gray")

        # 3. Rotation
        axes[row_idx, 2].imshow(rot_transform(raw_img), cmap="gray")

        # 4. Affine
        axes[row_idx, 3].imshow(affine_transform(raw_img), cmap="gray")

        # 5. Composite Model-Ready (Denormalized for visualization)
        tensor_sample = composite_transform(raw_img)  # [1, 48, 48] float32 normalized
        denorm_img = (
            denormalize(tensor_sample, mean=DEFAULT_TRAIN_MEAN, std=DEFAULT_TRAIN_STD)
            .squeeze()
            .numpy()
        )
        axes[row_idx, 5 - 1].imshow(denorm_img, cmap="gray", vmin=0.0, vmax=1.0)

        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(column_titles[col_idx], fontsize=11, fontweight="semibold", pad=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved preprocessing comparison grid to: %s", output_path)


def generate_stochastic_variations(
    record: DatasetRecord,
    output_path: Path,
    num_variants: int = 16,
) -> None:
    """Generate a 4x4 grid of stochastic variations on a single training face."""
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), dpi=300)
    fig.suptitle(
        f"FER2013 — Stochastic Training Augmentations (Class: {record.label_name.upper()})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    train_transform = build_train_transform()

    for i in range(num_variants):
        ax = axes[i // 4, i % 4]
        tensor_sample = train_transform(record.image)
        denorm_img = (
            denormalize(tensor_sample, mean=DEFAULT_TRAIN_MEAN, std=DEFAULT_TRAIN_STD)
            .squeeze()
            .numpy()
        )
        ax.imshow(denorm_img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Augment #{i + 1}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved stochastic variation grid to: %s", output_path)


def generate_normalization_distribution(
    records: list[DatasetRecord],
    output_path: Path,
) -> None:
    """Generate pixel distribution histograms for raw, scaled, and normalized data."""
    # Subsample 1,000 images for fast distribution plotting
    subsample = records[:1000]
    raw_pixels = np.concatenate([r.image.flatten() for r in subsample])
    scaled_pixels = raw_pixels / 255.0
    normalized_pixels = (scaled_pixels - DEFAULT_TRAIN_MEAN) / DEFAULT_TRAIN_STD

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    fig.suptitle(
        "FER2013 — Pixel Intensity Distribution Progression (Train Split)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # 1. Raw [0, 255]
    axes[0].hist(raw_pixels, bins=50, color="#3498db", edgecolor="black", alpha=0.8)
    axes[0].set_title("1. Raw Pixels [0, 255]", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Pixel Value")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Scaled [0.0, 1.0]
    axes[1].hist(scaled_pixels, bins=50, color="#2ecc71", edgecolor="black", alpha=0.8)
    axes[1].set_title("2. Scaled Pixels [0.0, 1.0]", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Float Value")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Normalized N(0, 1)
    axes[2].hist(normalized_pixels, bins=50, color="#e74c3c", edgecolor="black", alpha=0.8)
    axes[2].set_title(
        f"3. Normalized (mu={DEFAULT_TRAIN_MEAN:.3f}, sigma={DEFAULT_TRAIN_STD:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].set_xlabel("Normalized Value (Std Devs)")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved normalization distribution chart to: %s", output_path)


def main() -> None:
    set_seed(42)
    parser = argparse.ArgumentParser(
        description="Generate augmentation and preprocessing visualizations"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/fer2013/fer2013.csv",
        help="Path to raw FER2013 CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/interim/fer2013/preprocessing",
        help="Output directory for generated charts",
    )
    args = parser.parse_args()

    csv_path = ROOT_DIR / args.data_path
    out_dir = ROOT_DIR / args.output_dir

    if not csv_path.exists():
        logger.error("Dataset not found at: %s", csv_path)
        sys.exit(1)

    logger.info("Loading training records from %s...", csv_path)
    loader = FER2013Loader(csv_path)
    train_records = loader.load_records(split="train")

    logger.info("Generating preprocessing visual assets...")
    generate_preprocessing_comparison(train_records, out_dir / "preprocessing_comparison.png")

    # Pick a representative happy face for stochastic variations
    happy_samples = [r for r in train_records if r.label_name == "happy"]
    rep_sample = happy_samples[0] if happy_samples else train_records[0]
    generate_stochastic_variations(rep_sample, out_dir / "augmentation_samples.png")

    generate_normalization_distribution(train_records, out_dir / "normalization_distribution.png")

    logger.info("Visualizations successfully generated in: %s", out_dir)


if __name__ == "__main__":
    main()
