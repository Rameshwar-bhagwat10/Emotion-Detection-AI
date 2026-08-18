"""Confidence calculation and decision gating utilities."""

from __future__ import annotations

import numpy as np
import torch


def compute_confidence(probabilities: np.ndarray | torch.Tensor) -> float:
    """Compute confidence score as the maximum class probability.

    Args:
        probabilities: 1D probability distribution over classes.

    Returns:
        Confidence float in [0.0, 1.0].
    """
    if isinstance(probabilities, torch.Tensor):
        conf = float(torch.max(probabilities).item())
    else:
        conf = float(np.max(probabilities))

    return max(0.0, min(1.0, conf))


def apply_confidence_gate(
    emotion: str,
    confidence: float,
    threshold: float = 0.40,
    uncertain_label: str = "uncertain",
) -> tuple[str, bool]:
    """Apply application-level confidence thresholding gate.

    If model confidence is below threshold, marks decision as uncertain while
    preserving the underlying probability distribution.

    Args:
        emotion: Original predicted emotion string.
        confidence: Predicted class confidence in [0.0, 1.0].
        threshold: Confidence threshold below which prediction is marked uncertain.
        uncertain_label: Label assigned to uncertain predictions.

    Returns:
        Tuple of (effective_emotion, is_uncertain_boolean).
    """
    if confidence < threshold:
        return uncertain_label, True
    return emotion, False


def validate_probabilities(
    probabilities: np.ndarray | torch.Tensor,
    tolerance: float = 1e-3,
) -> bool:
    """Validate that probability values are non-negative, finite, and sum to approximately 1.0.

    Args:
        probabilities: 1D or 2D probability tensor/array.
        tolerance: Allowable tolerance for sum deviation from 1.0.

    Returns:
        Boolean indicating validity.
    """
    probs = (
        probabilities.detach().cpu().numpy()
        if isinstance(probabilities, torch.Tensor)
        else np.asarray(probabilities)
    )

    if not np.isfinite(probs).all():
        return False

    if (probs < -tolerance).any() or (probs > 1.0 + tolerance).any():
        return False

    if probs.ndim == 1:
        prob_sum = float(np.sum(probs))
        return abs(prob_sum - 1.0) <= tolerance
    elif probs.ndim == 2:
        prob_sums = np.sum(probs, axis=-1)
        return bool(np.all(np.abs(prob_sums - 1.0) <= tolerance))

    return False
