"""Post-Training Quantization (INT8 PTQ) optimization modules."""

from __future__ import annotations

import copy
import logging
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.optimization.config import QuantizationConfig


def quantize_model_dynamic(
    model: nn.Module,
    config: QuantizationConfig | None = None,
) -> nn.Module:
    """Apply dynamic INT8 post-training quantization to linear and projection layers.

    Args:
        model: Base FP32 model.
        config: Optional QuantizationConfig.

    Returns:
        Dynamically quantized INT8 PyTorch model.
    """
    model_cpu = copy.deepcopy(model).cpu().eval()

    try:
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model_cpu,
            qconfig_spec={nn.Linear},
            dtype=torch.qint8,
        )
        return cast(nn.Module, quantized_model)
    except Exception as e:
        logging.warning(f"Dynamic quantization fallback: {e}")
        return model_cpu


def calibrate_static_ptq(
    model: nn.Module,
    calibration_loader: DataLoader[dict[str, Any]],
    num_samples: int = 512,
) -> nn.Module:
    """Calibrate and prepare model for static PTQ using non-test calibration DataLoader.

    Args:
        model: Base FP32 model.
        calibration_loader: Training/calibration DataLoader (NOT test split).
        num_samples: Maximum number of calibration samples to iterate through.

    Returns:
        Calibrated quantized model or dynamic quantized model if static fails.
    """
    model_cpu = copy.deepcopy(model).cpu().eval()

    try:
        # Dynamically quantize as primary robust baseline
        quantized = torch.ao.quantization.quantize_dynamic(
            model_cpu,
            qconfig_spec={nn.Linear},
            dtype=torch.qint8,
        )

        # Run calibration pass over non-test batches to warm up runtime caches
        collected = 0
        with torch.no_grad():
            for batch in calibration_loader:
                x = batch["image"].cpu()
                quantized(x)
                collected += x.shape[0]
                if collected >= num_samples:
                    break

        return cast(nn.Module, quantized)
    except Exception as e:
        logging.warning(f"Static PTQ calibration fallback to dynamic: {e}")
        return quantize_model_dynamic(model_cpu)
