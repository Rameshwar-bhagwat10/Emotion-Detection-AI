"""Postprocessing engine converting raw model logits into structured emotion predictions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ml.datasets.fer2013.parser import EMOTION_NAMES
from ml.inference.confidence import (
    apply_confidence_gate,
    compute_confidence,
    validate_probabilities,
)
from ml.inference.face_detector import FaceDetection
from ml.inference.schemas import BoundingBoxDict, FacePrediction

DEFAULT_CLASSES = list(EMOTION_NAMES)

DEFAULT_CLASS_PRIORS: dict[str, float] = {
    "angry": 3995.0,
    "disgust": 436.0,
    "fear": 4097.0,
    "happy": 7215.0,
    "sad": 4965.0,
    "surprise": 3171.0,
    "neutral": 4965.0,
}


DEFAULT_CLASS_CALIBRATION_BIAS: dict[str, float] = {
    "angry": 0.0,
    "disgust": 0.0,
    "fear": 0.0,
    "happy": 0.0,
    "sad": 0.0,
    "surprise": 0.0,
    "neutral": 0.0,
}


def apply_logit_prior_adjustment(
    logits: torch.Tensor,
    classes: Sequence[str],
    tau: float = 0.30,
    class_priors: dict[str, float] | None = None,
    class_biases: dict[str, float] | None = None,
) -> torch.Tensor:
    """Apply balanced logit prior adjustment and calibration to eliminate class frequency bias.

    Formula:
        z_c* = z_c - tau * log(pi_c) + b_c

    When tau > 0, underrepresented classes (disgust, fear, surprise, angry, sad)
    receive an adjustment removing the high-frequency baseline of happy and neutral,
    and calibrated biases elevate sensitivity for subtle negative expressions.
    """
    adjusted = logits
    if tau > 0.0:
        priors_map = class_priors or DEFAULT_CLASS_PRIORS
        total = sum(priors_map.get(c, 1.0) for c in classes)
        priors_vec = [priors_map.get(c, 1.0) / total for c in classes]
        log_priors = torch.tensor(
            np.log(np.array(priors_vec, dtype=np.float32) + 1e-8),
            dtype=logits.dtype,
            device=logits.device,
        )
        adjusted = adjusted - tau * log_priors

    bias_map = class_biases if class_biases is not None else DEFAULT_CLASS_CALIBRATION_BIAS
    if bias_map:
        bias_vec = [bias_map.get(c, 0.0) for c in classes]
        bias_tensor = torch.tensor(bias_vec, dtype=logits.dtype, device=logits.device)
        adjusted = adjusted + bias_tensor

    return adjusted



def process_logits(
    logits: torch.Tensor,
    detections: Sequence[FaceDetection],
    classes: Sequence[str] | None = None,
    confidence_threshold: float = 0.40,
    uncertain_label: str = "uncertain",
    logit_adjustment_tau: float = 0.30,
    class_priors: dict[str, float] | None = None,
    class_biases: dict[str, float] | None = None,
) -> list[FacePrediction]:
    """Convert raw model logits into structured FacePrediction objects.

    Args:
        logits: Model output tensor of shape [B, num_classes].
        detections: Sequence of FaceDetection objects corresponding to each sample in the batch.
        classes: Ordered list of class names (defaults to authoritative FER-2013 classes).
        confidence_threshold: Confidence threshold for uncertainty gating.
        uncertain_label: String label for predictions below threshold.
        logit_adjustment_tau: Temperature for logit prior adjustment (0.0 to disable).
        class_priors: Optional dictionary mapping class names to prior frequencies.
        class_biases: Optional dictionary mapping class names to calibration additive biases.

    Returns:
        List of structured FacePrediction instances.

    Raises:
        ValueError: If logits shape does not match detections length or class count.
    """
    class_list = list(classes) if classes is not None else DEFAULT_CLASSES
    num_classes = len(class_list)

    if logits.ndim != 2:
        raise ValueError(
            f"Expected 2D logits tensor [B, {num_classes}], got shape {list(logits.shape)}"
        )
    if logits.shape[0] != len(detections):
        raise ValueError(
            f"Batch size mismatch: logits batch ({logits.shape[0]}) != detections count ({len(detections)})"
        )
    if logits.shape[1] != num_classes:
        raise ValueError(
            f"Logits dimension ({logits.shape[1]}) != expected class count ({num_classes})"
        )

    # Apply logit prior adjustment before softmax
    if logit_adjustment_tau > 0.0 or class_biases is not None:
        logits = apply_logit_prior_adjustment(
            logits,
            class_list,
            tau=logit_adjustment_tau,
            class_priors=class_priors,
            class_biases=class_biases,
        )

    # Apply softmax to obtain probability distributions
    probs_tensor = F.softmax(logits, dim=-1)
    probs_np = probs_tensor.detach().cpu().numpy()

    if not validate_probabilities(probs_np):
        raise ValueError("Invalid probability values detected after softmax execution.")

    predictions: list[FacePrediction] = []

    for i, detection in enumerate(detections):
        sample_probs = probs_np[i]
        predicted_idx = int(np.argmax(sample_probs))
        raw_emotion = class_list[predicted_idx]
        conf = compute_confidence(sample_probs)

        final_emotion, is_uncertain = apply_confidence_gate(
            emotion=raw_emotion,
            confidence=conf,
            threshold=confidence_threshold,
            uncertain_label=uncertain_label,
        )

        prob_dict = {class_list[c_idx]: float(sample_probs[c_idx]) for c_idx in range(num_classes)}

        bbox_dict = BoundingBoxDict(
            x=detection.bbox.x,
            y=detection.bbox.y,
            width=detection.bbox.width,
            height=detection.bbox.height,
        )

        predictions.append(
            FacePrediction(
                face_id=detection.face_id,
                bbox=bbox_dict,
                detection_confidence=detection.confidence,
                emotion=final_emotion,
                confidence=conf,
                is_uncertain=is_uncertain,
                probabilities=prob_dict,
            )
        )

    return predictions
