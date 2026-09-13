"""Unit tests for TemporalSmoother service in the FastAPI application."""

import pytest
from app.services.temporal_smoother import TemporalSmoother


def test_temporal_smoother_initialization():
    """Verify first frame sets initial emotion and probability distribution."""
    smoother = TemporalSmoother(confidence_threshold=0.40, probability_margin=0.03)
    raw_probs = {
        "angry": 0.05,
        "disgust": 0.02,
        "fear": 0.03,
        "happy": 0.10,
        "sad": 0.10,
        "surprise": 0.15,
        "neutral": 0.55,
    }
    emotion, conf, is_uncertain, smoothed_probs = smoother.smooth(
        face_id=1,
        raw_probabilities=raw_probs,
        raw_emotion="neutral",
        raw_confidence=0.55,
    )

    assert emotion == "neutral"
    assert conf == 0.55
    assert not is_uncertain
    assert smoothed_probs["neutral"] == 0.55


def test_temporal_smoother_responsive_neutral_exit():
    """Verify face transitions away from neutral when an active emotion emerges."""
    smoother = TemporalSmoother(confidence_threshold=0.40, probability_margin=0.03)
    
    # Step 1: Initialized as neutral
    neutral_probs = {
        "angry": 0.05, "disgust": 0.02, "fear": 0.03, "happy": 0.05,
        "sad": 0.10, "surprise": 0.05, "neutral": 0.70,
    }
    smoother.smooth(1, neutral_probs, "neutral", 0.70)

    # Step 2: Sudden Angry expression arrives (e.g. angry: 0.55, neutral: 0.20)
    angry_probs = {
        "angry": 0.65, "disgust": 0.05, "fear": 0.05, "happy": 0.02,
        "sad": 0.05, "surprise": 0.03, "neutral": 0.15,
    }
    emotion, conf, is_uncertain, smoothed_probs = smoother.smooth(1, angry_probs, "angry", 0.65)

    # Thanks to responsive neutral exit, it transitions without staying locked on neutral
    assert emotion == "angry"
    assert smoothed_probs["angry"] > smoothed_probs["neutral"]


def test_temporal_smoother_uncertain_gate():
    """Verify low confidence predictions are flagged as uncertain."""
    smoother = TemporalSmoother(confidence_threshold=0.40)
    flat_probs = {
        "angry": 0.15, "disgust": 0.14, "fear": 0.14, "happy": 0.14,
        "sad": 0.14, "surprise": 0.15, "neutral": 0.14,
    }
    emotion, conf, is_uncertain, _ = smoother.smooth(2, flat_probs, "angry", 0.15)
    assert emotion == "uncertain"
    assert is_uncertain


def test_temporal_smoother_pruning():
    """Verify inactive face states are pruned after max_missing_steps."""
    smoother = TemporalSmoother(max_missing_steps=3)
    probs = {"angry": 0.1, "disgust": 0.1, "fear": 0.1, "happy": 0.1, "sad": 0.1, "surprise": 0.1, "neutral": 0.4}
    smoother.smooth(10, probs, "neutral", 0.4)
    assert 10 in smoother._face_states

    # Frame with face missing
    for _ in range(4):
        smoother.prune_missing(active_face_ids=set())

    assert 10 not in smoother._face_states
