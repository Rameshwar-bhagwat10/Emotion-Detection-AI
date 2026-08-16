"""Hardware and compute device detection utilities."""

from typing import Any

import torch


def get_device(preferred: str = "auto") -> torch.device:
    """Select and return appropriate PyTorch device based on preference and hardware availability.

    Args:
        preferred: 'auto', 'cuda', 'mps', or 'cpu'.

    Returns:
        torch.device instance.
    """
    pref = preferred.lower().strip()

    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif pref == "cpu":
        return torch.device("cpu")

    # Auto detection
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> dict[str, Any]:
    """Retrieve detailed hardware information for compute devices.

    Returns:
        Dictionary containing device type, availability, GPU name, and memory stats.
    """
    device = get_device("auto")
    info: dict[str, Any] = {
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": None,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
        info["memory_reserved_mb"] = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)

    return info
