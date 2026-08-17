"""Confusion matrix computation, normalization, pair extraction, and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def compute_confusion_matrix(
    y_true: np.ndarray | torch.Tensor | list[int],
    y_pred: np.ndarray | torch.Tensor | list[int],
    num_classes: int = 7,
) -> np.ndarray:
    """Compute raw confusion matrix where rows are actual and columns are predicted.

    Args:
        y_true: Ground truth integer class indices.
        y_pred: Predicted integer class indices.
        num_classes: Total number of classes.

    Returns:
        np.ndarray of shape (num_classes, num_classes) with integer sample counts.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_t = np.asarray(y_true, dtype=np.int64).flatten()
    y_p = np.asarray(y_pred, dtype=np.int64).flatten()

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_t, y_p, strict=True):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
        else:
            raise ValueError(f"Label out of bounds [0, {num_classes-1}]: actual={t}, pred={p}")

    return cm


def compute_normalized_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    """Compute normalized confusion matrix where each row sums to 1.0 (or 0 if support is 0).

    Args:
        cm: Raw confusion matrix of shape (N, N).

    Returns:
        np.ndarray of shape (N, N) with float proportions in [0.0, 1.0].
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero for classes with 0 support
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_cm = np.divide(cm.astype(np.float64), row_sums)
        norm_cm = np.nan_to_num(norm_cm, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(norm_cm, dtype=np.float64)


def extract_confusion_pairs(
    cm: np.ndarray,
    class_names: list[str],
) -> list[dict[str, Any]]:
    """Extract and sort all off-diagonal confusion pairs in descending order of frequency.

    Args:
        cm: Raw confusion matrix.
        class_names: Ordered list of class names.

    Returns:
        List of dicts: [{'actual_class': ..., 'predicted_class': ..., 'count': ...}, ...]
    """
    num_classes = len(class_names)
    pairs: list[dict[str, Any]] = []

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {
                        "actual_class": class_names[i],
                        "predicted_class": class_names[j],
                        "count": int(cm[i, j]),
                    }
                )

    # Sort descending by count
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs


def save_confusion_matrix_csv(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save confusion matrix as a labeled CSV file.

    Args:
        cm: Confusion matrix array.
        class_names: Column/Index emotion labels.
        output_path: Output CSV file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.index.name = "actual"
    df.to_csv(output_path)


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    normalized: bool = False,
    title: str | None = None,
) -> None:
    """Render and save a high-resolution confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (raw counts or normalized float).
        class_names: Class labels.
        output_path: PNG file destination.
        normalized: Whether values represent normalized ratios.
        title: Plot title.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = ".2f" if normalized else "d"
    plot_title = title or ("Normalized Confusion Matrix" if normalized else "Confusion Matrix")

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title(plot_title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Emotion", fontsize=12, labelpad=8)
    plt.ylabel("Actual Emotion", fontsize=12, labelpad=8)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
