"""Unit tests for postprocessing, softmax probability calculation, and confidence gating."""

from __future__ import annotations

import numpy as np
import torch

from ml.inference.confidence import (
    apply_confidence_gate,
    compute_confidence,
    validate_probabilities,
)
from ml.inference.face_detector import FaceBoundingBox, FaceDetection
from ml.inference.postprocessing import process_logits


def test_compute_confidence():
    """Verify confidence score is the maximum probability value."""
    probs = np.array([0.05, 0.05, 0.10, 0.70, 0.05, 0.03, 0.02], dtype=np.float32)
    conf = compute_confidence(probs)
    assert abs(conf - 0.70) < 1e-5


def test_apply_confidence_gate():
    """Verify confidence gating above and below threshold."""
    # Above threshold (0.65 >= 0.40) -> returns predicted emotion
    em_high, is_unc_high = apply_confidence_gate(
        emotion="happy",
        confidence=0.65,
        threshold=0.40,
        uncertain_label="uncertain",
    )
    assert em_high == "happy"
    assert is_unc_high is False

    # Below threshold (0.35 < 0.40) -> returns uncertain label
    em_low, is_unc_low = apply_confidence_gate(
        emotion="sad",
        confidence=0.35,
        threshold=0.40,
        uncertain_label="uncertain",
    )
    assert em_low == "uncertain"
    assert is_unc_low is True


def test_validate_probabilities():
    """Verify validation of probability vectors."""
    valid_probs = np.array([0.1, 0.2, 0.1, 0.3, 0.1, 0.1, 0.1])
    assert validate_probabilities(valid_probs) is True

    # Sum does not equal 1.0
    invalid_sum = np.array([0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert validate_probabilities(invalid_sum) is False

    # Contains NaN
    nan_probs = np.array([0.5, np.nan, 0.2, 0.1, 0.1, 0.05, 0.05])
    assert validate_probabilities(nan_probs) is False


def test_process_logits_single_and_batch():
    """Verify process_logits converts raw logits to structured FacePredictions."""
    classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    # Create synthetic logits: Sample 0 has highest on 'happy' (idx 3), Sample 1 has highest on 'surprise' (idx 5)
    logits = torch.tensor(
        [
            [-1.0, -2.0, -1.0, 5.0, -1.0, -1.0, -1.0],
            [-2.0, -2.0, -2.0, -1.0, -1.0, 4.0, -2.0],
        ],
        dtype=torch.float32,
    )

    detections = [
        FaceDetection(
            face_id=1, bbox=FaceBoundingBox(x=10, y=20, width=50, height=50), confidence=0.95
        ),
        FaceDetection(
            face_id=2, bbox=FaceBoundingBox(x=70, y=30, width=60, height=60), confidence=0.90
        ),
    ]

    preds = process_logits(
        logits=logits,
        detections=detections,
        classes=classes,
        confidence_threshold=0.40,
    )

    assert len(preds) == 2
    assert preds[0].face_id == 1
    assert preds[0].emotion == "happy"
    assert preds[0].confidence > 0.90
    assert preds[0].is_uncertain is False
    assert abs(sum(preds[0].probabilities.values()) - 1.0) < 1e-4

    assert preds[1].face_id == 2
    assert preds[1].emotion == "surprise"
    assert preds[1].confidence > 0.85
    assert preds[1].is_uncertain is False
