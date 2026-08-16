"""Tests for compute device detection and hardware reporting."""

import torch

from ml.utils.device import get_device, get_device_info


def test_get_device_auto():
    """Verify get_device returns a valid torch.device instance."""
    device = get_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in ["cpu", "cuda", "mps"]


def test_get_device_explicit_cpu():
    """Verify explicit CPU selection."""
    device = get_device("cpu")
    assert device.type == "cpu"


def test_get_device_info():
    """Verify get_device_info returns structured dictionary with hardware stats."""
    info = get_device_info()
    assert isinstance(info, dict)
    assert "device_type" in info
    assert "cuda_available" in info
    assert "mps_available" in info
    assert "device_count" in info
    assert isinstance(info["cuda_available"], bool)
