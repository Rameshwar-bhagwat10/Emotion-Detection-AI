"""Random seed management utilities for reproducible experiments."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic execution.

    Args:
        seed: The integer seed value to apply.
        deterministic: If True, configures PyTorch and cuDNN for strict determinism.

    Returns:
        The seed integer that was applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    return seed


def get_seed_from_env(default: int = 42) -> int:
    """Retrieve seed from environment variable SEED or return default."""
    seed_str = os.getenv("SEED")
    if seed_str and seed_str.isdigit():
        return int(seed_str)
    return default
