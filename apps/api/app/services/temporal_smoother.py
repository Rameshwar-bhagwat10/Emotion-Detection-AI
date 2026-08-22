"""Temporal emotion probability smoother using Exponential Moving Average (EMA)."""

from __future__ import annotations

from typing import Any


class TemporalSmoother:
    """Maintains per-face smoothed probability distributions across consecutive video frames.

    Applies Exponential Moving Average (EMA):
        P_smooth[t] = alpha * P_raw[t] + (1 - alpha) * P_smooth[t-1]
    """

    def __init__(
        self,
        alpha: float = 0.6,
        confidence_threshold: float = 0.40,
        max_missing_steps: int = 15,
    ) -> None:
        """Initialize TemporalSmoother.

        Args:
            alpha: Weight for current raw prediction (higher = more responsive, lower = smoother).
            confidence_threshold: Threshold below which dominant emotion is labeled 'uncertain'.
            max_missing_steps: Steps before an inactive tracked face state is purged.
        """
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.max_missing_steps = max_missing_steps
        # Mapping: track_id -> {"probs": dict[str, float], "missing": int}
        self._face_states: dict[int, dict[str, Any]] = {}

    def smooth(
        self,
        face_id: int,
        raw_probabilities: dict[str, float],
        raw_emotion: str,
        raw_confidence: float,
    ) -> tuple[str, float, bool, dict[str, float]]:
        """Apply EMA smoothing to a face's probability distribution.

        Args:
            face_id: Track ID of the detected face.
            raw_probabilities: Raw 7-class probability map from Phase 09 inference.
            raw_emotion: Raw dominant emotion.
            raw_confidence: Raw dominant confidence score.

        Returns:
            Tuple of `(smoothed_emotion, smoothed_confidence, is_uncertain, smoothed_probabilities)`.
        """
        if face_id not in self._face_states:
            # First time seeing this face: initialize with raw probabilities
            self._face_states[face_id] = {
                "probs": dict(raw_probabilities),
                "missing": 0,
            }
            smoothed_probs = dict(raw_probabilities)
        else:
            state = self._face_states[face_id]
            state["missing"] = 0
            prev_probs = state["probs"]

            smoothed_probs = {}
            for emotion_cls, raw_p in raw_probabilities.items():
                prev_p = prev_probs.get(emotion_cls, raw_p)
                smoothed_probs[emotion_cls] = self.alpha * raw_p + (1.0 - self.alpha) * prev_p

            # Normalize probabilities so they sum to 1.0
            total_sum = sum(smoothed_probs.values())
            if total_sum > 0:
                smoothed_probs = {k: round(v / total_sum, 4) for k, v in smoothed_probs.items()}

            state["probs"] = smoothed_probs

        # Determine dominant smoothed emotion
        dominant_emotion = max(smoothed_probs, key=lambda k: smoothed_probs[k])
        dominant_confidence = float(smoothed_probs[dominant_emotion])

        is_uncertain = dominant_confidence < self.confidence_threshold
        final_emotion = "uncertain" if is_uncertain else dominant_emotion

        return final_emotion, dominant_confidence, is_uncertain, smoothed_probs

    def prune_missing(self, active_face_ids: set[int]) -> None:
        """Increment missing counter for faces not in active_face_ids and prune expired tracks."""
        to_remove: list[int] = []
        for face_id, state in self._face_states.items():
            if face_id not in active_face_ids:
                state["missing"] += 1
                if state["missing"] > self.max_missing_steps:
                    to_remove.append(face_id)
        for face_id in to_remove:
            del self._face_states[face_id]

    def reset(self) -> None:
        """Purge all face smoothing history."""
        self._face_states.clear()
