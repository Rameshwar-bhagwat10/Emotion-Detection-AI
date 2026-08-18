"""Confidence analysis, error analysis, and prediction recording utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def analyze_confidence(
    y_true: np.ndarray | torch.Tensor | list[int],
    y_pred: np.ndarray | torch.Tensor | list[int],
    probabilities: np.ndarray | torch.Tensor,
    class_names: list[str],
    high_confidence_threshold: float = 0.90,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Perform comprehensive confidence analysis and high-confidence error extraction.

    Args:
        y_true: True class indices (1D).
        y_pred: Predicted class indices (1D).
        probabilities: Softmax probability distribution (2D, shape [N, num_classes]).
        class_names: Ordered emotion class names.
        high_confidence_threshold: Threshold above which errors are flagged as high-confidence.
        num_bins: Histogram bin count for confidence distribution.

    Returns:
        Structured analysis dictionary.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    y_t = np.asarray(y_true, dtype=np.int64).flatten()
    y_p = np.asarray(y_pred, dtype=np.int64).flatten()
    probs = np.asarray(probabilities, dtype=np.float64)

    if len(y_t) != len(y_p) or len(y_t) != len(probs):
        raise ValueError("Length mismatch between y_true, y_pred, and probabilities")

    confidences = np.max(probs, axis=-1)
    correct_mask = y_t == y_p
    incorrect_mask = ~correct_mask

    correct_confs = confidences[correct_mask]
    incorrect_confs = confidences[incorrect_mask]

    # Global summary
    total_samples = len(y_t)
    total_correct = int(np.sum(correct_mask))
    total_incorrect = int(np.sum(incorrect_mask))

    # High confidence errors
    high_conf_error_mask = incorrect_mask & (confidences >= high_confidence_threshold)
    high_conf_error_indices = np.where(high_conf_error_mask)[0]

    high_conf_errors: list[dict[str, Any]] = []
    for idx in high_conf_error_indices:
        high_conf_errors.append(
            {
                "sample_id": int(idx),
                "actual_class": class_names[y_t[idx]],
                "predicted_class": class_names[y_p[idx]],
                "confidence": round(float(confidences[idx]), 6),
            }
        )

    # All incorrect predictions
    incorrect_records: list[dict[str, Any]] = []
    for idx in np.where(incorrect_mask)[0]:
        incorrect_records.append(
            {
                "sample_id": int(idx),
                "actual_class": class_names[y_t[idx]],
                "predicted_class": class_names[y_p[idx]],
                "confidence": round(float(confidences[idx]), 6),
            }
        )

    # Histogram binning [0.0, 1.0]
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    hist_all, _ = np.histogram(confidences, bins=bins)
    hist_correct, _ = (
        np.histogram(correct_confs, bins=bins)
        if len(correct_confs) > 0
        else (np.zeros(num_bins, dtype=int), bins)
    )
    hist_incorrect, _ = (
        np.histogram(incorrect_confs, bins=bins)
        if len(incorrect_confs) > 0
        else (np.zeros(num_bins, dtype=int), bins)
    )

    distribution = []
    for b_idx in range(num_bins):
        distribution.append(
            {
                "bin_range": f"{bins[b_idx]:.2f}-{bins[b_idx+1]:.2f}",
                "total_count": int(hist_all[b_idx]),
                "correct_count": int(hist_correct[b_idx]),
                "incorrect_count": int(hist_incorrect[b_idx]),
            }
        )

    # Per-class confidence breakdown
    class_conf_stats: dict[str, dict[str, float]] = {}
    for c_idx, c_name in enumerate(class_names):
        c_mask = y_t == c_idx
        if np.any(c_mask):
            c_confs = confidences[c_mask]
            class_conf_stats[c_name] = {
                "mean_confidence": round(float(np.mean(c_confs)), 6),
                "median_confidence": round(float(np.median(c_confs)), 6),
                "std_confidence": round(float(np.std(c_confs)), 6),
            }
        else:
            class_conf_stats[c_name] = {
                "mean_confidence": 0.0,
                "median_confidence": 0.0,
                "std_confidence": 0.0,
            }

    return {
        "overall": {
            "total_samples": total_samples,
            "mean_confidence": round(float(np.mean(confidences)), 6) if total_samples > 0 else 0.0,
            "median_confidence": (
                round(float(np.median(confidences)), 6) if total_samples > 0 else 0.0
            ),
            "std_confidence": round(float(np.std(confidences)), 6) if total_samples > 0 else 0.0,
        },
        "correct_predictions": {
            "count": total_correct,
            "mean_confidence": (
                round(float(np.mean(correct_confs)), 6) if len(correct_confs) > 0 else 0.0
            ),
            "median_confidence": (
                round(float(np.median(correct_confs)), 6) if len(correct_confs) > 0 else 0.0
            ),
            "std_confidence": (
                round(float(np.std(correct_confs)), 6) if len(correct_confs) > 0 else 0.0
            ),
        },
        "incorrect_predictions": {
            "count": total_incorrect,
            "mean_confidence": (
                round(float(np.mean(incorrect_confs)), 6) if len(incorrect_confs) > 0 else 0.0
            ),
            "median_confidence": (
                round(float(np.median(incorrect_confs)), 6) if len(incorrect_confs) > 0 else 0.0
            ),
            "std_confidence": (
                round(float(np.std(incorrect_confs)), 6) if len(incorrect_confs) > 0 else 0.0
            ),
        },
        "high_confidence_errors": {
            "threshold": high_confidence_threshold,
            "count": len(high_conf_errors),
            "percentage_of_errors": (
                round(float(len(high_conf_errors) / total_incorrect * 100), 2)
                if total_incorrect > 0
                else 0.0
            ),
            "samples": high_conf_errors,
        },
        "confidence_distribution": distribution,
        "class_confidence": class_conf_stats,
        "all_incorrect_samples": incorrect_records,
    }


def save_incorrect_predictions_csv(
    incorrect_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save all misclassified samples to a structured CSV file.

    Args:
        incorrect_records: List of misclassification dictionaries.
        output_path: Destination CSV file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(incorrect_records)
    df.to_csv(output_path, index=False)


def save_confidence_analysis_json(
    analysis: dict[str, Any],
    output_path: Path,
) -> None:
    """Serialize confidence analysis (excluding verbose raw sample dump) to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save a clean copy without massive raw array duplicates
    clean_dict = {k: v for k, v in analysis.items() if k != "all_incorrect_samples"}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_dict, f, indent=2)


def plot_and_save_confidence_distribution(
    correct_confs: np.ndarray,
    incorrect_confs: np.ndarray,
    output_path: Path,
    num_bins: int = 10,
) -> None:
    """Generate and save dual-histogram visualization of prediction confidences.

    Args:
        correct_confs: 1D array of correct prediction confidences.
        incorrect_confs: 1D array of incorrect prediction confidences.
        output_path: Destination PNG path.
        num_bins: Number of histogram bins.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bins_list = [float(b) for b in np.linspace(0.0, 1.0, num_bins + 1)]

    plt.figure(figsize=(9, 5))
    if len(correct_confs) > 0:
        plt.hist(
            correct_confs,
            bins=bins_list,
            alpha=0.6,
            color="#2ecc71",
            label=f"Correct (N={len(correct_confs)})",
            edgecolor="black",
            linewidth=0.8,
        )
    if len(incorrect_confs) > 0:
        plt.hist(
            incorrect_confs,
            bins=bins_list,
            alpha=0.6,
            color="#e74c3c",
            label=f"Incorrect (N={len(incorrect_confs)})",
            edgecolor="black",
            linewidth=0.8,
        )

    plt.title(
        "Prediction Confidence Distribution (Correct vs Incorrect)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Confidence (Maximum Softmax Probability)", fontsize=11, labelpad=8)
    plt.ylabel("Sample Count", fontsize=11, labelpad=8)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def compute_expected_calibration_error(
    y_true: np.ndarray | torch.Tensor | list[int],
    probabilities: np.ndarray | torch.Tensor,
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins.

    Args:
        y_true: True class labels (1D).
        probabilities: Softmax probabilities (2D, shape [N, num_classes]).
        num_bins: Number of confidence bins (default: 10).

    Returns:
        ECE value in range [0.0, 1.0].
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    y_t = np.asarray(y_true, dtype=np.int64).flatten()
    probs = np.asarray(probabilities, dtype=np.float64)

    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = predictions == y_t

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total_samples = len(y_t)

    if total_samples == 0:
        return 0.0

    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        in_bin = (
            (confidences > bin_lower) & (confidences <= bin_upper)
            if i > 0
            else (confidences >= bin_lower) & (confidences <= bin_upper)
        )
        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(round(ece, 6))
