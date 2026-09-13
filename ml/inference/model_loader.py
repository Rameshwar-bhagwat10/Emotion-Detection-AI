"""Model loading, lifecycle management, device placement, and warm-up service."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ml.inference.config import ModelInferenceConfig
from ml.models.factory import create_model


class ModelLoadingError(Exception):
    """Base exception for model loading failures."""

    pass


def resolve_device(strategy: str = "auto", allow_fallback_to_cpu: bool = True) -> torch.device:
    """Resolve torch device based on configuration strategy.

    Args:
        strategy: 'auto', 'cpu', or 'cuda'.
        allow_fallback_to_cpu: Whether to fallback to CPU if CUDA is requested but unavailable.

    Returns:
        torch.device instance.
    """
    strat = strategy.lower().strip()
    if strat == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif allow_fallback_to_cpu:
            logging.warning("CUDA requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        else:
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    elif strat == "cpu":
        return torch.device("cpu")
    elif strat == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError(f"Unknown device strategy '{strategy}'. Must be 'auto', 'cpu', or 'cuda'.")


def _read_metadata_file(meta_path_str: str) -> dict[str, Any]:
    """Read metadata dictionary from path."""
    meta_path = Path(meta_path_str)
    if not meta_path.is_absolute():
        meta_path = Path(__file__).resolve().parent.parent.parent / meta_path_str

    if not meta_path.exists():
        raise ModelLoadingError(f"Optimized Champion metadata file not found at {meta_path}")

    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
            return dict(data) if isinstance(data, dict) else {}
    except Exception as e:
        raise ModelLoadingError(f"Failed to read Champion metadata: {e}") from e


def _load_state_dict_weights(weights_path_str: str) -> dict[str, Any]:
    """Load state dictionary weights from checkpoint."""
    weights_path = Path(weights_path_str)
    if not weights_path.is_absolute():
        weights_path = Path(__file__).resolve().parent.parent.parent / weights_path_str

    if not weights_path.exists():
        raise ModelLoadingError(f"Optimized Champion weights file not found at {weights_path}")

    # Detect un-pulled Git LFS pointers (< 1KB containing 'git-lfs' or 'version https')
    # and automatically hydrate real weights from GitHub Media CDN
    if weights_path.stat().st_size < 1000:
        try:
            with open(weights_path, "r", errors="ignore") as f:
                header = f.read(200)
            if "git-lfs" in header or "version https" in header:
                logging.info(
                    f"Detected un-pulled Git LFS pointer at {weights_path}. Hydrating from GitHub Media CDN..."
                )
                import urllib.request
                media_url = (
                    "https://media.githubusercontent.com/media/Rameshwar-bhagwat10/"
                    "Emotion-Detection-AI/main/artifacts/optimized/champion/model.pt"
                )
                req = urllib.request.Request(
                    media_url,
                    headers={"User-Agent": "Mozilla/5.0 (EmotionDetectionAI-Hydrator/1.0)"},
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    blob = resp.read()
                    if len(blob) > 1000:
                        with open(weights_path, "wb") as out_f:
                            out_f.write(blob)
                        logging.info(
                            f"Successfully hydrated model weights from CDN ({len(blob)} bytes)."
                        )
                    else:
                        raise ModelLoadingError(f"Hydration payload unexpectedly small: {len(blob)} bytes.")
        except Exception as hyd_err:
            logging.error(f"Failed to auto-hydrate Champion model weights: {hyd_err}")
            raise ModelLoadingError(
                f"Weights file at {weights_path} is an un-pulled Git LFS pointer and CDN hydration failed: {hyd_err}"
            ) from hyd_err

    try:
        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            return checkpoint
        raise ValueError("Unexpected checkpoint payload type.")
    except Exception as e:
        raise ModelLoadingError(f"Failed to load state dict from {weights_path}: {e}") from e


class ModelManager:
    """Manages Champion model lifecycle, caching, warm-up, and device placement."""

    _instance: ModelManager | None = None

    def __init__(
        self, config: ModelInferenceConfig | None = None, device: torch.device | None = None
    ) -> None:
        """Initialize ModelManager with configuration."""
        self.config = config or ModelInferenceConfig()
        self.device = device or resolve_device()
        self.model: nn.Module | None = None
        self.metadata: dict[str, Any] = {}
        self._is_warmed_up = False

    def load_champion(self) -> tuple[nn.Module, dict[str, Any]]:
        """Load and cache the Phase 08 Optimized Champion PyTorch model.

        Returns:
            Tuple of (loaded_model, metadata_dictionary).
        """
        if self.model is not None:
            return self.model, self.metadata

        self.metadata = _read_metadata_file(self.config.metadata_file)
        base_arch = self.metadata.get("architecture") or self.metadata.get("base_model") or self.config.expected_architecture

        try:
            model = create_model(base_arch)
        except Exception as e:
            raise ModelLoadingError(f"Failed to instantiate architecture '{base_arch}': {e}") from e

        state_dict = _load_state_dict_weights(self.config.weights_file)
        model.load_state_dict(state_dict)
        model.to(device=self.device)
        model.eval()

        if self.device.type == "cpu":
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)

        self.model = model
        logging.info(
            f"Loaded Optimized Champion '{self.metadata.get('model_name')}' ({base_arch}) onto device '{self.device}' (threads={torch.get_num_threads()})."
        )
        return self.model, self.metadata

    def warmup(self, iterations: int = 3) -> None:
        """Run non-gradient warm-up forward passes to prime memory, CUDA kernels, and caches.

        Args:
            iterations: Number of warm-up iterations.
        """
        if self.model is None:
            self.load_champion()

        if iterations <= 0 or self._is_warmed_up or self.model is None:
            return

        h, w = self.config.input_size
        c = self.config.input_channels
        dummy_tensor = torch.randn(1, c, h, w, dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            for _ in range(iterations):
                _ = self.model(dummy_tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        self._is_warmed_up = True
        logging.info(f"Completed {iterations} warm-up iterations on device '{self.device}'.")


def load_champion_model(
    config: ModelInferenceConfig | None = None,
    device: torch.device | None = None,
    warmup_iterations: int = 3,
) -> tuple[nn.Module, dict[str, Any], torch.device]:
    """Convenience function to load and optionally warm up the Optimized Champion.

    Args:
        config: Optional ModelInferenceConfig.
        device: Optional target device.
        warmup_iterations: Warm-up passes.

    Returns:
        Tuple of (model, metadata, device).
    """
    manager = ModelManager(config=config, device=device)
    model, metadata = manager.load_champion()
    if warmup_iterations > 0:
        manager.warmup(warmup_iterations)
    return model, metadata, manager.device
