"""Reduced precision (FP16 / BF16) optimization utilities."""

from __future__ import annotations

import copy
from typing import cast

import numpy as np
import torch
from torch import nn


class ReducedPrecisionWrapper(nn.Module):
    """Wrapper executing underlying model in half or bfloat16 precision."""

    def __init__(self, model: nn.Module, dtype: torch.dtype = torch.float16) -> None:
        """Initialize Reduced Precision Wrapper.

        Args:
            model: Base PyTorch neural network.
            dtype: Target precision dtype (torch.float16 or torch.bfloat16).
        """
        super().__init__()
        self.model = copy.deepcopy(model)
        self.dtype = dtype

        # Convert parameters to target dtype if supported
        try:
            self.model.to(dtype=self.dtype)
        except Exception:
            # Fallback to maintaining FP32 weights if direct casting fails on specific hardware
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass casting inputs and outputs appropriately.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Logits in standard float32 representation for downstream loss/metrics consistency.
        """
        orig_dtype = x.dtype
        x_in = x.to(dtype=self.dtype)
        out = self.model(x_in)
        return cast(torch.Tensor, out.to(dtype=orig_dtype))


def evaluate_numerical_difference(
    base_model: nn.Module,
    optimized_model: nn.Module,
    sample_input: torch.Tensor,
) -> dict[str, float]:
    """Compute numerical divergence metrics between base and optimized models.

    Args:
        base_model: Reference FP32 model.
        optimized_model: Optimized model.
        sample_input: Test tensor [B, C, H, W].

    Returns:
        Dictionary containing max_absolute_error, mean_absolute_error, and cosine_similarity.
    """
    base_model.eval()
    optimized_model.eval()

    with torch.no_grad():
        out_base = base_model(sample_input).detach().cpu().numpy()
        out_opt = optimized_model(sample_input).detach().cpu().numpy()

    abs_diff = np.abs(out_base - out_opt)
    max_err = float(np.max(abs_diff))
    mean_err = float(np.mean(abs_diff))

    # Cosine similarity across outputs
    flat_base = out_base.flatten()
    flat_opt = out_opt.flatten()
    denom = np.linalg.norm(flat_base) * np.linalg.norm(flat_opt)
    cos_sim = float(np.dot(flat_base, flat_opt) / (denom + 1e-12))

    return {
        "max_absolute_error": round(max_err, 6),
        "mean_absolute_error": round(mean_err, 6),
        "cosine_similarity": round(cos_sim, 6),
    }
