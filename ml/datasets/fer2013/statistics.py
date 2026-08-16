"""FER2013 Dataset Statistics, Duplicate/Leakage Analysis, and EDA Visualizer.

Provides comprehensive statistical computation, cross-split duplicate/leakage detection,
and reproducible visualization generation for exploratory data analysis (EDA).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ml.datasets.fer2013.parser import (
    EMOTION_NAMES,
    DatasetRecord,
)

logger = logging.getLogger(__name__)

# Premium aesthetic color palette for emotion charts
EMOTION_PALETTE = {
    "angry": "#EF4444",  # Red
    "disgust": "#10B981",  # Emerald green
    "fear": "#8B5CF6",  # Purple
    "happy": "#F59E0B",  # Amber/Gold
    "sad": "#3B82F6",  # Blue
    "surprise": "#EC4899",  # Pink
    "neutral": "#6B7280",  # Cool slate gray
}


@dataclass
class DuplicateAnalysisResult:
    """Outcome of duplicate and cross-split data leakage analysis."""

    total_samples: int = 0
    unique_samples: int = 0
    exact_duplicates_count: int = 0
    duplicate_groups_count: int = 0
    duplicates_within_train: int = 0
    duplicates_within_val: int = 0
    duplicates_within_test: int = 0
    train_val_overlap: int = 0
    train_test_overlap: int = 0
    val_test_overlap: int = 0
    cross_split_leakage_detected: bool = False
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class DatasetStatisticsResult:
    """Comprehensive statistical summary of FER2013 dataset."""

    total_samples: int = 0
    num_classes: int = 7
    classes: dict[str, int] = field(default_factory=dict)
    class_percentages: dict[str, float] = field(default_factory=dict)
    class_imbalance_ratio: float = 1.0
    most_represented_class: tuple[str, int] = ("", 0)
    least_represented_class: tuple[str, int] = ("", 0)
    splits: dict[str, int] = field(default_factory=dict)
    split_percentages: dict[str, float] = field(default_factory=dict)
    split_class_distributions: dict[str, dict[str, int]] = field(default_factory=dict)
    image_dimensions: dict[str, Any] = field(default_factory=dict)
    pixel_statistics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class FER2013Statistics:
    """Statistical analyzer and EDA generator for FER2013."""

    def __init__(self) -> None:
        """Initialize statistics generator."""
        sns.set_theme(style="darkgrid", palette="muted")

    def compute_statistics(self, records: list[DatasetRecord]) -> DatasetStatisticsResult:
        """Compute full statistical metrics from a list of DatasetRecords.

        Args:
            records: List of parsed DatasetRecord objects.

        Returns:
            DatasetStatisticsResult: Computed metrics.
        """
        if not records:
            return DatasetStatisticsResult()

        total = len(records)
        class_counts: dict[str, int] = dict.fromkeys(EMOTION_NAMES, 0)
        split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        split_classes: dict[str, dict[str, int]] = {
            "train": dict.fromkeys(EMOTION_NAMES, 0),
            "val": dict.fromkeys(EMOTION_NAMES, 0),
            "test": dict.fromkeys(EMOTION_NAMES, 0),
        }

        min_w, max_w = 48, 48
        min_h, max_h = 48, 48
        channels = 1

        pixel_sum: float = 0.0
        pixel_sq_sum: float = 0.0
        total_pixel_count: int = 0
        p_min = 255
        p_max = 0

        for r in records:
            # Class tally
            class_counts[r.label_name] = class_counts.get(r.label_name, 0) + 1

            # Split tally
            split_counts[r.split] = split_counts.get(r.split, 0) + 1
            if r.split in split_classes:
                split_classes[r.split][r.label_name] = (
                    split_classes[r.split].get(r.label_name, 0) + 1
                )

            # Dimensions
            h, w = r.shape
            min_h = min(min_h, h)
            max_h = max(max_h, h)
            min_w = min(min_w, w)
            max_w = max(max_w, w)

            # Pixel stats
            img_arr = r.image.astype(np.float64)
            p_min = min(p_min, int(img_arr.min()))
            p_max = max(p_max, int(img_arr.max()))
            pixel_sum += float(img_arr.sum())
            pixel_sq_sum += float((img_arr**2).sum())
            total_pixel_count += img_arr.size

        # Class percentages
        class_pcts = {k: round((v / total) * 100, 2) for k, v in class_counts.items()}
        split_pcts = {k: round((v / total) * 100, 2) for k, v in split_counts.items()}

        # Imbalance
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
        least_rep = sorted_classes[0] if sorted_classes else ("", 0)
        most_rep = sorted_classes[-1] if sorted_classes else ("", 0)
        imbalance_ratio = round(most_rep[1] / max(1, least_rep[1]), 2)

        # Pixel mean and std
        p_mean = round(pixel_sum / max(1, total_pixel_count), 2)
        p_var = max(0.0, (pixel_sq_sum / max(1, total_pixel_count)) - (p_mean**2))
        p_std = round(float(np.sqrt(p_var)), 2)

        return DatasetStatisticsResult(
            total_samples=total,
            num_classes=len(EMOTION_NAMES),
            classes=class_counts,
            class_percentages=class_pcts,
            class_imbalance_ratio=imbalance_ratio,
            most_represented_class=most_rep,
            least_represented_class=least_rep,
            splits=split_counts,
            split_percentages=split_pcts,
            split_class_distributions=split_classes,
            image_dimensions={
                "min_width": min_w,
                "max_width": max_w,
                "min_height": min_h,
                "max_height": max_h,
                "channels": channels,
                "uniform_48x48": (min_w == max_w == min_h == max_h == 48),
            },
            pixel_statistics={
                "min": float(p_min),
                "max": float(p_max),
                "mean": p_mean,
                "std": p_std,
            },
        )

    def compute_duplicates(  # noqa: C901
        self, records: list[DatasetRecord]
    ) -> DuplicateAnalysisResult:
        """Perform exact pixel hash duplicate analysis and cross-split leakage audit.

        Args:
            records: List of parsed DatasetRecord objects.

        Returns:
            DuplicateAnalysisResult: Detailed duplicate and leakage metrics.
        """
        hash_map: dict[str, list[DatasetRecord]] = defaultdict(list)
        for r in records:
            hash_map[r.md5_hash].append(r)

        exact_duplicates_count = 0
        duplicate_groups_count = 0
        dups_train = 0
        dups_val = 0
        dups_test = 0
        train_val_overlap = 0
        train_test_overlap = 0
        val_test_overlap = 0
        overlap_details: list[dict[str, Any]] = []

        for h, group in hash_map.items():
            if len(group) > 1:
                duplicate_groups_count += 1
                exact_duplicates_count += len(group) - 1

                splits = {rec.split for rec in group}
                train_count = sum(1 for rec in group if rec.split == "train")
                val_count = sum(1 for rec in group if rec.split == "val")
                test_count = sum(1 for rec in group if rec.split == "test")

                if train_count > 1:
                    dups_train += train_count - 1
                if val_count > 1:
                    dups_val += val_count - 1
                if test_count > 1:
                    dups_test += test_count - 1

                # Cross split overlap
                has_train_val = ("train" in splits) and ("val" in splits)
                has_train_test = ("train" in splits) and ("test" in splits)
                has_val_test = ("val" in splits) and ("test" in splits)

                if has_train_val:
                    train_val_overlap += 1
                if has_train_test:
                    train_test_overlap += 1
                if has_val_test:
                    val_test_overlap += 1

                if len(splits) > 1:
                    overlap_details.append(
                        {
                            "hash": h,
                            "count": len(group),
                            "splits": list(splits),
                            "labels": list({rec.label_name for rec in group}),
                        }
                    )

        cross_split_detected = (
            train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0
        )

        return DuplicateAnalysisResult(
            total_samples=len(records),
            unique_samples=len(hash_map),
            exact_duplicates_count=exact_duplicates_count,
            duplicate_groups_count=duplicate_groups_count,
            duplicates_within_train=dups_train,
            duplicates_within_val=dups_val,
            duplicates_within_test=dups_test,
            train_val_overlap=train_val_overlap,
            train_test_overlap=train_test_overlap,
            val_test_overlap=val_test_overlap,
            cross_split_leakage_detected=cross_split_detected,
            details=overlap_details[:50],  # cap at first 50 for report brevity
        )

    def generate_charts(self, stats: DatasetStatisticsResult, output_dir: Path) -> None:
        """Generate high-resolution EDA visual charts and save to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Class Distribution Bar Chart
        plt.figure(figsize=(10, 6), dpi=300)
        classes = list(stats.classes.keys())
        counts = list(stats.classes.values())
        colors = [EMOTION_PALETTE.get(c, "#3B82F6") for c in classes]

        bars = plt.bar(classes, counts, color=colors, edgecolor="#1F2937", linewidth=1.2)
        plt.title(
            "FER2013 - Overall Emotion Class Distribution", fontsize=14, fontweight="bold", pad=15
        )
        plt.xlabel("Emotion Class", fontsize=12, labelpad=10)
        plt.ylabel("Sample Count", fontsize=12, labelpad=10)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            yval = bar.get_height()
            pct = round((yval / max(1, stats.total_samples)) * 100, 1)
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + max(counts) * 0.015,
                f"{yval:,}\n({pct}%)",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="semibold",
            )

        plt.ylim(0, max(counts) * 1.15)
        plt.tight_layout()
        plt.savefig(output_dir / "class_distribution.png")
        plt.close()

        # 2. Split Distribution Grouped Bar Chart
        plt.figure(figsize=(12, 6), dpi=300)
        x = np.arange(len(classes))
        width = 0.25

        train_counts = [stats.split_class_distributions.get("train", {}).get(c, 0) for c in classes]
        val_counts = [stats.split_class_distributions.get("val", {}).get(c, 0) for c in classes]
        test_counts = [stats.split_class_distributions.get("test", {}).get(c, 0) for c in classes]

        plt.bar(x - width, train_counts, width, label="Train", color="#3B82F6", edgecolor="#1E3A8A")
        plt.bar(x, val_counts, width, label="Validation", color="#10B981", edgecolor="#065F46")
        plt.bar(x + width, test_counts, width, label="Test", color="#F59E0B", edgecolor="#92400E")

        plt.title(
            "FER2013 - Emotion Distribution Across Train / Val / Test Splits",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Emotion Class", fontsize=12, labelpad=10)
        plt.ylabel("Sample Count", fontsize=12, labelpad=10)
        plt.xticks(x, classes)
        plt.legend(frameon=True, facecolor="white", loc="upper right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "split_distribution.png")
        plt.close()

        # 3. Pixel Statistics / Intensity Distribution Chart
        plt.figure(figsize=(10, 5), dpi=300)
        p_stats = stats.pixel_statistics
        plt.axvline(
            p_stats.get("mean", 128.0),
            color="#EF4444",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {p_stats.get('mean')}",
        )
        plt.axvspan(
            max(0, p_stats.get("mean", 128.0) - p_stats.get("std", 50.0)),
            min(255, p_stats.get("mean", 128.0) + p_stats.get("std", 50.0)),
            color="#3B82F6",
            alpha=0.2,
            label=f"±1 Std ({p_stats.get('std')})",
        )
        plt.xlim(0, 255)
        plt.title(
            "FER2013 - Pixel Intensity Distribution Summary", fontsize=14, fontweight="bold", pad=15
        )
        plt.xlabel("Grayscale Pixel Intensity (0 - 255)", fontsize=12)
        plt.ylabel("Density Representation", fontsize=12)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(output_dir / "pixel_distribution.png")
        plt.close()

        # 4. Image Dimensions Summary Chart
        plt.figure(figsize=(8, 4), dpi=300)
        dims = stats.image_dimensions
        plt.text(
            0.5,
            0.5,
            f"Image Dimensions Audit\n\n"
            f"Width: {dims.get('min_width')}px - {dims.get('max_width')}px\n"
            f"Height: {dims.get('min_height')}px - {dims.get('max_height')}px\n"
            f"Channels: {dims.get('channels')} (Grayscale)\n"
            f"Uniform 48x48: {'YES (100% compliant)' if dims.get('uniform_48x48') else 'NO'}",
            ha="center",
            va="center",
            fontsize=13,
            bbox={
                "boxstyle": "round,pad=1",
                "facecolor": "#EFF6FF",
                "edgecolor": "#3B82F6",
                "linewidth": 2,
            },
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / "image_dimensions.png")
        plt.close()

        logger.info("Saved EDA charts to %s", output_dir)

    def generate_sample_grids(  # noqa: C901
        self,
        records: list[DatasetRecord],
        output_dir: Path,
        seed: int = 42,
    ) -> None:
        """Generate reproducible sample image grids and class visualizations.

        Args:
            records: List of parsed DatasetRecord objects.
            output_dir: Destination folder.
            seed: Deterministic random seed.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)

        # 1. Class-wise sample grid (7 rows x 5 columns = 35 images)
        fig, axes = plt.subplots(7, 5, figsize=(10, 14), dpi=300)
        fig.suptitle(
            "FER2013 - Representative Class-Wise Samples",
            fontsize=16,
            fontweight="bold",
            y=0.99,
        )

        by_class: dict[str, list[DatasetRecord]] = defaultdict(list)
        for r in records:
            by_class[r.label_name].append(r)

        for row_idx, emotion in enumerate(EMOTION_NAMES):
            class_records = by_class.get(emotion, [])
            selected_records: list[DatasetRecord]
            if len(class_records) >= 5:
                selected_records = list(rng.choice(class_records, size=5, replace=False))  # type: ignore[arg-type]
            else:
                selected_records = class_records[:5]

            for col_idx, rec in enumerate(selected_records):
                ax = axes[row_idx, col_idx]
                ax.imshow(rec.image, cmap="gray", vmin=0, vmax=255)
                ax.axis("off")
                if col_idx == 0:
                    ax.set_title(
                        f"{emotion.upper()}",
                        fontsize=11,
                        fontweight="bold",
                        loc="left",
                        color=EMOTION_PALETTE.get(emotion, "black"),
                    )

        plt.tight_layout()
        plt.savefig(output_dir / "class_samples.png")
        plt.close()

        # 2. Random 5x5 grid
        fig, axes = plt.subplots(5, 5, figsize=(10, 10), dpi=300)
        fig.suptitle(
            "FER2013 - Deterministic Random Sample Grid (Seed 42)",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        rand_records: list[DatasetRecord]
        if len(records) >= 25:
            rand_records = list(rng.choice(records, size=25, replace=False))  # type: ignore[arg-type]
        else:
            rand_records = records[:25]

        for idx, rec in enumerate(rand_records):
            ax = axes[idx // 5, idx % 5]
            ax.imshow(rec.image, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{rec.label_name} ({rec.split})", fontsize=9, fontweight="semibold")
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / "random_samples.png")
        plt.close()

        # 3. Class-specific sample images (e.g. angry_samples.png, etc.)
        for emotion in EMOTION_NAMES:
            class_records = by_class.get(emotion, [])
            if not class_records:
                continue
            count = min(6, len(class_records))
            selected = rng.choice(class_records, size=count, replace=False)  # type: ignore[arg-type]
            fig, axes = plt.subplots(1, count, figsize=(count * 2.2, 2.6), dpi=250)
            if count == 1:
                axes = [axes]  # type: ignore[assignment]
            fig.suptitle(f"FER2013 - {emotion.upper()} Samples", fontsize=12, fontweight="bold")
            for ax, rec in zip(axes, selected, strict=False):
                ax.imshow(rec.image, cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"ID #{rec.record_id}", fontsize=8)
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(output_dir / f"{emotion}_samples.png")
            plt.close()

        logger.info("Saved sample grids to %s", output_dir)
