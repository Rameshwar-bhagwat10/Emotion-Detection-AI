"""Tests for reproducibility and random seed utilities."""

import random

import numpy as np
import torch

from ml.utils.reproducibility import setup_reproducibility
from ml.utils.seed import set_seed


def test_set_seed():
    """Verify set_seed enforces deterministic random sequence generation."""
    set_seed(1234)
    val_py1 = random.random()
    val_np1 = np.random.rand()
    val_torch1 = torch.rand(1).item()

    set_seed(1234)
    val_py2 = random.random()
    val_np2 = np.random.rand()
    val_torch2 = torch.rand(1).item()

    assert val_py1 == val_py2
    assert val_np1 == val_np2
    assert val_torch1 == val_torch2


def test_setup_reproducibility():
    """Verify setup_reproducibility returns configuration dictionary."""
    config = setup_reproducibility(seed=42, deterministic=True)
    assert config["seed"] == 42
    assert config["deterministic"] is True
    assert "device_info" in config
