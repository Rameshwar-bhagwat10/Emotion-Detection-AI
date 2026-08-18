"""Deployment export utilities (PyTorch -> ONNX) and ONNX Runtime validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from ml.optimization.config import ExportConfig


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    config: ExportConfig | None = None,
    input_shape: tuple[int, int, int, int] = (1, 1, 48, 48),
) -> Path:
    """Export a PyTorch neural network to standard ONNX format.

    Args:
        model: PyTorch model to export.
        output_path: Target path for .onnx file.
        config: Optional ExportConfig instance.
        input_shape: Model input shape tuple [B, C, H, W] (default: [1, 1, 48, 48]).

    Returns:
        Path to exported ONNX model artifact.
    """
    cfg = config or ExportConfig()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = model.cpu().eval()
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    dynamic_axes = (
        {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
        if cfg.dynamic_axes
        else None
    )

    try:
        torch.onnx.export(
            model_cpu,
            (dummy_input,),
            str(out_p),
            export_params=True,
            opset_version=cfg.opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    except TypeError:
        # For PyTorch versions that do not accept dynamo argument
        torch.onnx.export(
            model_cpu,
            (dummy_input,),
            str(out_p),
            export_params=True,
            opset_version=cfg.opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    # Validate exported ONNX model graph structure
    onnx_model = onnx.load(str(out_p))
    onnx.checker.check_model(onnx_model)

    return out_p


def validate_onnx_export(
    onnx_path: str | Path,
    pytorch_model: nn.Module,
    atol: float = 1e-3,
    num_samples: int = 8,
) -> dict[str, Any]:
    """Verify that ONNX Runtime inference produces outputs consistent with PyTorch.

    Args:
        onnx_path: Path to exported .onnx artifact.
        pytorch_model: Ground truth PyTorch model.
        atol: Maximum allowed absolute numerical tolerance.
        num_samples: Number of random verification samples.

    Returns:
        Validation dictionary containing max_error, mean_error, prediction_match_rate, and is_valid.
    """
    p_path = Path(onnx_path)
    if not p_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {p_path}")

    pytorch_model = pytorch_model.cpu().eval()

    # Create ONNX Runtime Session
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(p_path), session_options, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    dummy_inputs = torch.randn(num_samples, 1, 48, 48, dtype=torch.float32)

    with torch.no_grad():
        torch_outputs = pytorch_model(dummy_inputs).numpy()

    onnx_inputs = {input_name: dummy_inputs.numpy()}
    onnx_outputs = session.run([output_name], onnx_inputs)[0]

    abs_diff = np.abs(torch_outputs - onnx_outputs)
    max_error = float(np.max(abs_diff))
    mean_error = float(np.mean(abs_diff))

    torch_preds = np.argmax(torch_outputs, axis=-1)
    onnx_preds = np.argmax(onnx_outputs, axis=-1)
    match_rate = float(np.mean(torch_preds == onnx_preds))

    is_valid = max_error <= atol and match_rate == 1.0

    return {
        "onnx_path": str(p_path),
        "total_verification_samples": num_samples,
        "max_absolute_error": round(max_error, 6),
        "mean_absolute_error": round(mean_error, 6),
        "prediction_match_rate": round(match_rate, 4),
        "numerical_tolerance": atol,
        "is_valid": is_valid,
    }
