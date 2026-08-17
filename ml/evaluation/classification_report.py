"""Classification report generation and serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.evaluation.metrics import calculate_all_metrics


def generate_classification_report(
    y_true: Any,
    y_pred: Any,
    class_names: list[str],
) -> dict[str, Any]:
    """Generate structured classification report with per-class and aggregate metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Ordered list of emotion names.

    Returns:
        Structured dictionary matching classification report schema.
    """
    metrics = calculate_all_metrics(y_true, y_pred, class_names)
    total_support = sum(int(v["support"]) for v in metrics["per_class"].values())

    report: dict[str, Any] = {
        "accuracy": metrics["accuracy"],
        "macro_avg": {
            "precision": metrics["macro_precision"],
            "recall": metrics["macro_recall"],
            "f1-score": metrics["macro_f1"],
            "support": total_support,
        },
        "weighted_avg": {
            "precision": metrics["weighted_precision"],
            "recall": metrics["weighted_recall"],
            "f1-score": metrics["weighted_f1"],
            "support": total_support,
        },
        "classes": metrics["per_class"],
    }
    return report


def save_classification_report_json(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Serialize classification report to formatted JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def save_classification_report_csv(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Serialize classification report to CSV table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    # Per-class rows
    for class_name, metrics in report.get("classes", {}).items():
        rows.append(
            {
                "entity": class_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1-score": metrics["f1"],
                "support": metrics["support"],
            }
        )

    # Macro and Weighted avg rows
    macro = report.get("macro_avg", {})
    rows.append(
        {
            "entity": "macro_avg",
            "precision": macro.get("precision", 0.0),
            "recall": macro.get("recall", 0.0),
            "f1-score": macro.get("f1-score", 0.0),
            "support": macro.get("support", 0),
        }
    )

    weighted = report.get("weighted_avg", {})
    rows.append(
        {
            "entity": "weighted_avg",
            "precision": weighted.get("precision", 0.0),
            "recall": weighted.get("recall", 0.0),
            "f1-score": weighted.get("f1-score", 0.0),
            "support": weighted.get("support", 0),
        }
    )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def format_classification_report_table(report: dict[str, Any]) -> str:
    """Format classification report as a human-readable ASCII table."""
    lines: list[str] = [
        f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}",
        "-" * 56,
    ]

    for c_name, c_met in report.get("classes", {}).items():
        lines.append(
            f"{c_name:<12} {c_met['precision']:>10.4f} {c_met['recall']:>10.4f} "
            f"{c_met['f1']:>10.4f} {c_met['support']:>10d}"
        )

    lines.append("-" * 56)
    lines.append(
        f"{'Accuracy':<12} {'':>10} {'':>10} {report.get('accuracy', 0.0):>10.4f} {report.get('macro_avg', {}).get('support', 0):>10d}"
    )

    macro = report.get("macro_avg", {})
    lines.append(
        f"{'Macro Avg':<12} {macro.get('precision', 0.0):>10.4f} {macro.get('recall', 0.0):>10.4f} "
        f"{macro.get('f1-score', 0.0):>10.4f} {macro.get('support', 0):>10d}"
    )

    weighted = report.get("weighted_avg", {})
    lines.append(
        f"{'Weighted Avg':<12} {weighted.get('precision', 0.0):>10.4f} {weighted.get('recall', 0.0):>10.4f} "
        f"{weighted.get('f1-score', 0.0):>10.4f} {weighted.get('support', 0):>10d}"
    )

    return "\n".join(lines)
