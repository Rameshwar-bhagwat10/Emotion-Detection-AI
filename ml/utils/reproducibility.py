"""Reproducibility configuration utilities for deep learning pipelines."""

import os
from typing import Any

from ml.utils.device import get_device_info
from ml.utils.seed import set_seed


def setup_reproducibility(seed: int = 42, deterministic: bool = True) -> dict[str, Any]:
    """Configure entire execution environment for reproducible training and inference.

    Args:
        seed: Random seed integer.
        deterministic: Whether to enforce deterministic algorithm execution.

    Returns:
        Dictionary summarizing reproducibility configuration and device information.
    """
    applied_seed = set_seed(seed=seed, deterministic=deterministic)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    device_info = get_device_info()

    return {
        "seed": applied_seed,
        "deterministic": deterministic,
        "device_info": device_info,
    }
