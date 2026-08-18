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


def process_logits(
    logits: torch.Tensor,
    detections: Sequence[FaceDetection],
    classes: Sequence[str] | None = None,
    confidence_threshold: float = 0.40,
    uncertain_label: str = "uncertain",
) -> list[FacePrediction]:
    """Convert raw model logits into structured FacePrediction objects.

    Args:
        logits: Model output tensor of shape [B, num_classes].
        detections: Sequence of FaceDetection objects corresponding to each sample in the batch.
        classes: Ordered list of class names (defaults to authoritative FER-2013 classes).
        confidence_threshold: Confidence threshold for uncertainty gating.
        uncertain_label: String label for predictions below threshold.

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
