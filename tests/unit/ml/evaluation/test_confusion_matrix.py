"""Unit tests for confusion matrix computation, normalization, and pair extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.evaluation.confusion_matrix import (
    compute_confusion_matrix,
    compute_normalized_confusion_matrix,
    extract_confusion_pairs,
    plot_and_save_confusion_matrix,
    save_confusion_matrix_csv,
)


def test_compute_confusion_matrix_shape_and_values() -> None:
    """Verify raw confusion matrix values."""
    # 0 -> 0 (2 times), 0 -> 1 (1 time)
    # 1 -> 1 (3 times)
    # 2 -> 0 (1 time), 2 -> 2 (2 times)
    y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 1, 0, 2, 2]

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
    assert cm.shape == (3, 3)
    assert np.array_equal(
        cm,
        np.array(
            [
                [2, 1, 0],
                [0, 3, 0],
                [1, 0, 2],
            ]
        ),
    )


def test_compute_normalized_confusion_matrix() -> None:
    """Verify normalized confusion matrix rows sum to 1.0."""
    cm = np.array(
        [
            [2, 2],
            [1, 3],
        ],
        dtype=np.int64,
    )
    norm_cm = compute_normalized_confusion_matrix(cm)
    assert norm_cm.shape == (2, 2)
    assert np.allclose(norm_cm.sum(axis=1), np.ones(2))
    assert norm_cm[0, 0] == 0.5 and norm_cm[0, 1] == 0.5
    assert norm_cm[1, 0] == 0.25 and norm_cm[1, 1] == 0.75


def test_extract_confusion_pairs_sorted() -> None:
    """Verify off-diagonal confusion pairs sorted descending."""
    cm = np.array(
        [
            [10, 5, 2],
            [1, 20, 8],
            [0, 3, 30],
        ]
    )
    class_names = ["angry", "disgust", "fear"]
    pairs = extract_confusion_pairs(cm, class_names)

    # Expected off-diagonal pairs: 5 pairs sorted descending by count
    assert len(pairs) == 5
    assert pairs[0] == {"actual_class": "disgust", "predicted_class": "fear", "count": 8}
    assert pairs[1] == {"actual_class": "angry", "predicted_class": "disgust", "count": 5}
    assert pairs[2] == {"actual_class": "fear", "predicted_class": "disgust", "count": 3}
    assert pairs[3] == {"actual_class": "angry", "predicted_class": "fear", "count": 2}
    assert pairs[4] == {"actual_class": "disgust", "predicted_class": "angry", "count": 1}


def test_save_csv_and_plot(tmp_path: Path) -> None:
    """Verify CSV writing and PNG plotting."""
    cm = np.array([[5, 1], [2, 4]])
    class_names = ["class_0", "class_1"]

    csv_path = tmp_path / "confusion_matrix.csv"
    save_confusion_matrix_csv(cm, class_names, csv_path)
    assert csv_path.exists()
    assert "class_0,5,1" in csv_path.read_text(encoding="utf-8")

    png_path = tmp_path / "confusion_matrix.png"
    plot_and_save_confusion_matrix(cm, class_names, png_path)
    assert png_path.exists()
    assert png_path.stat().st_size > 0
