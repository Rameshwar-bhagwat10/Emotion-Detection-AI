"""Unit tests for classification report generation and formatting."""

from __future__ import annotations

import json
from pathlib import Path

from ml.evaluation.classification_report import (
    format_classification_report_table,
    generate_classification_report,
    save_classification_report_csv,
    save_classification_report_json,
)


def test_generate_classification_report_structure() -> None:
    """Verify dictionary structure of generated report."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    class_names = ["class_a", "class_b"]

    report = generate_classification_report(y_true, y_pred, class_names)

    assert "accuracy" in report
    assert "macro_avg" in report
    assert "weighted_avg" in report
    assert "classes" in report
    assert "class_a" in report["classes"] and "class_b" in report["classes"]


def test_save_report_json_and_csv(tmp_path: Path) -> None:
    """Verify JSON and CSV saving."""
    y_true = [0, 1]
    y_pred = [0, 1]
    class_names = ["c0", "c1"]
    report = generate_classification_report(y_true, y_pred, class_names)

    json_path = tmp_path / "report.json"
    save_classification_report_json(report, json_path)
    assert json_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["accuracy"] == 1.0

    csv_path = tmp_path / "report.csv"
    save_classification_report_csv(report, csv_path)
    assert csv_path.exists()
    csv_content = csv_path.read_text(encoding="utf-8")
    assert "c0" in csv_content and "macro_avg" in csv_content


def test_format_table_string() -> None:
    """Verify human readable ASCII table string output."""
    y_true = [0, 1]
    y_pred = [0, 1]
    class_names = ["happy", "sad"]
    report = generate_classification_report(y_true, y_pred, class_names)
    table_str = format_classification_report_table(report)

    assert "Class" in table_str
    assert "happy" in table_str
    assert "sad" in table_str
    assert "Macro Avg" in table_str
