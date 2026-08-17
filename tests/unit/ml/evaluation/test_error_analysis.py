"""Unit tests for confidence analysis and error tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.evaluation.error_analysis import (
    analyze_confidence,
    plot_and_save_confidence_distribution,
    save_incorrect_predictions_csv,
)


def test_analyze_confidence_high_confidence_error_detection() -> None:
    """Verify high-confidence misclassifications are correctly identified."""
    class_names = ["angry", "happy"]
    y_true = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 0]  # Sample 1 and 3 are incorrect

    # Probabilities:
    # Sample 0: [0.8, 0.2] -> Correct, conf 0.8
    # Sample 1: [0.95, 0.05] -> Incorrect (true=1, pred=0), conf 0.95 (HIGH CONF ERROR >= 0.90)
    # Sample 2: [0.7, 0.3] -> Correct, conf 0.7
    # Sample 3: [0.6, 0.4] -> Incorrect (true=1, pred=0), conf 0.60
    probabilities = np.array(
        [
            [0.80, 0.20],
            [0.95, 0.05],
            [0.70, 0.30],
            [0.60, 0.40],
        ]
    )

    analysis = analyze_confidence(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_names=class_names,
        high_confidence_threshold=0.90,
    )

    assert analysis["overall"]["total_samples"] == 4
    assert analysis["correct_predictions"]["count"] == 2
    assert analysis["incorrect_predictions"]["count"] == 2
    assert analysis["high_confidence_errors"]["count"] == 1
    assert analysis["high_confidence_errors"]["samples"][0]["sample_id"] == 1
    assert analysis["high_confidence_errors"]["samples"][0]["confidence"] == 0.95


def test_save_incorrect_and_plot(tmp_path: Path) -> None:
    """Verify CSV and PNG persistence."""
    records = [
        {"sample_id": 1, "actual_class": "angry", "predicted_class": "sad", "confidence": 0.92}
    ]
    csv_path = tmp_path / "incorrect.csv"
    save_incorrect_predictions_csv(records, csv_path)
    assert csv_path.exists()
    assert "angry,sad,0.92" in csv_path.read_text(encoding="utf-8")

    png_path = tmp_path / "conf_dist.png"
    plot_and_save_confidence_distribution(
        correct_confs=np.array([0.9, 0.85]),
        incorrect_confs=np.array([0.6, 0.75]),
        output_path=png_path,
    )
    assert png_path.exists()
    assert png_path.stat().st_size > 0
