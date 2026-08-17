"""Evaluation metrics computation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def calculate_accuracy(
    y_true: np.ndarray | torch.Tensor | list[int],
    y_pred: np.ndarray | torch.Tensor | list[int],
) -> float:
    """Calculate overall classification accuracy.

    Args:
        y_true: Ground truth labels (1D).
        y_pred: Predicted class labels (1D).

    Returns:
        float: Accuracy in range [0.0, 1.0].
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_t = np.asarray(y_true, dtype=np.int64).flatten()
    y_p = np.asarray(y_pred, dtype=np.int64).flatten()

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")
    if len(y_t) == 0:
        return 0.0

    return float(np.mean(y_t == y_p))


def calculate_per_class_metrics(
    y_true: np.ndarray | torch.Tensor | list[int],
    y_pred: np.ndarray | torch.Tensor | list[int],
    num_classes: int = 7,
    class_names: list[str] | None = None,
    zero_division: float = 0.0,
) -> dict[str, dict[str, float | int]]:
    """Compute per-class precision, recall, F1-score, and support.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted class labels.
        num_classes: Number of classes.
        class_names: Optional class names mapping.
        zero_division: Fallback value when denominator is 0.

    Returns:
        Dictionary mapping class_name -> {'precision': ..., 'recall': ..., 'f1': ..., 'support': ...}.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_t = np.asarray(y_true, dtype=np.int64).flatten()
    y_p = np.asarray(y_pred, dtype=np.int64).flatten()

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    elif len(class_names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, got {len(class_names)}")

    per_class: dict[str, dict[str, float | int]] = {}

    for c_idx, c_name in enumerate(class_names):
        tp = int(np.sum((y_t == c_idx) & (y_p == c_idx)))
        fp = int(np.sum((y_t != c_idx) & (y_p == c_idx)))
        fn = int(np.sum((y_t == c_idx) & (y_p != c_idx)))
        support = int(np.sum(y_t == c_idx))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else zero_division
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else zero_division
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else zero_division
        )

        per_class[c_name] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    return per_class


def calculate_macro_metrics(
    per_class_metrics: dict[str, dict[str, float | int]],
) -> dict[str, float]:
    """Compute unweighted macro-averaged precision, recall, and F1.

    Args:
        per_class_metrics: Per-class metrics dictionary.

    Returns:
        Dictionary with macro_precision, macro_recall, macro_f1.
    """
    if not per_class_metrics:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    precisions = [float(v["precision"]) for v in per_class_metrics.values()]
    recalls = [float(v["recall"]) for v in per_class_metrics.values()]
    f1s = [float(v["f1"]) for v in per_class_metrics.values()]

    return {
        "macro_precision": round(float(np.mean(precisions)), 6),
        "macro_recall": round(float(np.mean(recalls)), 6),
        "macro_f1": round(float(np.mean(f1s)), 6),
    }


def calculate_weighted_metrics(
    per_class_metrics: dict[str, dict[str, float | int]],
) -> dict[str, float]:
    """Compute support-weighted precision, recall, and F1.

    Args:
        per_class_metrics: Per-class metrics dictionary.

    Returns:
        Dictionary with weighted_precision, weighted_recall, weighted_f1.
    """
    total_support = sum(int(v["support"]) for v in per_class_metrics.values())
    if total_support == 0:
        return {"weighted_precision": 0.0, "weighted_recall": 0.0, "weighted_f1": 0.0}

    w_prec = sum(float(v["precision"]) * int(v["support"]) for v in per_class_metrics.values())
    w_rec = sum(float(v["recall"]) * int(v["support"]) for v in per_class_metrics.values())
    w_f1 = sum(float(v["f1"]) * int(v["support"]) for v in per_class_metrics.values())

    return {
        "weighted_precision": round(w_prec / total_support, 6),
        "weighted_recall": round(w_rec / total_support, 6),
        "weighted_f1": round(w_f1 / total_support, 6),
    }


def calculate_all_metrics(
    y_true: np.ndarray | torch.Tensor | list[int],
    y_pred: np.ndarray | torch.Tensor | list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """Compute comprehensive suite of classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted class labels.
        class_names: Canonical class names.

    Returns:
        Combined metrics dictionary including overall, macro, weighted, and per-class metrics.
    """
    acc = calculate_accuracy(y_true, y_pred)
    per_class = calculate_per_class_metrics(
        y_true, y_pred, num_classes=len(class_names), class_names=class_names
    )
    macro = calculate_macro_metrics(per_class)
    weighted = calculate_weighted_metrics(per_class)

    return {
        "accuracy": round(acc, 6),
        "macro_precision": macro["macro_precision"],
        "macro_recall": macro["macro_recall"],
        "macro_f1": macro["macro_f1"],
        "weighted_precision": weighted["weighted_precision"],
        "weighted_recall": weighted["weighted_recall"],
        "weighted_f1": weighted["weighted_f1"],
        "per_class": per_class,
    }
