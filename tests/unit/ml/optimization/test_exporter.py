"""Unit tests for ONNX model export and ONNX Runtime validation."""

from __future__ import annotations

from pathlib import Path

from ml.models.factory import create_model
from ml.optimization.exporter import export_to_onnx, validate_onnx_export


def test_export_to_onnx_and_validate(tmp_path: Path) -> None:
    """Verify PyTorch model exports to ONNX and passes ONNX Runtime validation."""
    model = create_model("baseline_cnn")
    onnx_path = tmp_path / "model.onnx"

    exported_p = export_to_onnx(model, onnx_path)
    assert exported_p.exists()
    assert exported_p.stat().st_size > 1000

    report = validate_onnx_export(onnx_path, model, atol=1e-3, num_samples=4)
    assert report["is_valid"] is True
    assert report["prediction_match_rate"] == 1.0
    assert report["max_absolute_error"] <= 1e-3
