"""Lightweight multi-face spatial tracking and session-local ID persistence."""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.schemas.prediction import BoundingBoxSchema


@dataclass
class TrackedFace:
    """Internal state for a single tracked face."""

    track_id: int
    bbox: BoundingBoxSchema
    centroid: tuple[float, float]
    frames_since_seen: int = 0
    total_detections: int = 1


def calculate_iou(box_a: BoundingBoxSchema, box_b: BoundingBoxSchema) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box_a: First bounding box.
        box_b: Second bounding box.

    Returns:
        IoU overlap ratio in [0.0, 1.0].
    """
    x_a = max(box_a.x, box_b.x)
    y_a = max(box_a.y, box_b.y)
    x_b = min(box_a.x + box_a.width, box_b.x + box_b.width)
    y_b = min(box_a.y + box_a.height, box_b.y + box_b.height)

    inter_width = max(0, x_b - x_a)
    inter_height = max(0, y_b - y_a)
    inter_area = inter_width * inter_height

    box_a_area = box_a.width * box_a.height
    box_b_area = box_b.width * box_b.height

    union_area = box_a_area + box_b_area - inter_area
    if union_area <= 0:
        return 0.0

    return float(inter_area) / float(union_area)


def calculate_centroid(box: BoundingBoxSchema) -> tuple[float, float]:
    """Compute Euclidean centroid coordinate (cx, cy) of a bounding box."""
    return (box.x + box.width / 2.0, box.y + box.height / 2.0)


def calculate_centroid_distance(c1: tuple[float, float], c2: tuple[float, float]) -> float:
    """Compute Euclidean distance between two centroids."""
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


class FaceTracker:
    """Lightweight session-local multi-face spatial tracker.

    Uses greedy Intersection over Union (IoU) and Centroid distance heuristics
    to associate detected face boxes across consecutive video frames without
    biometric identification or face recognition models.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_centroid_distance: float = 80.0,
        max_frames_missing: int = 15,
    ) -> None:
        """Initialize FaceTracker parameters.

        Args:
            iou_threshold: Minimum IoU overlap required to consider boxes a match.
            max_centroid_distance: Max Euclidean pixel distance for centroid matching.
            max_frames_missing: Number of frames before a lost track is pruned.
        """
        self.iou_threshold = iou_threshold
        self.max_centroid_distance = max_centroid_distance
        self.max_frames_missing = max_frames_missing

        self._next_track_id: int = 1
        self._active_tracks: dict[int, TrackedFace] = {}

    def _prune_missing_tracks(self) -> None:
        """Increment missing count and delete stale tracks."""
        to_remove = [
            t_id
            for t_id, track in self._active_tracks.items()
            if track.frames_since_seen > self.max_frames_missing
        ]
        for t_id in to_remove:
            del self._active_tracks[t_id]

    def _match_iou(
        self,
        detected_bboxes: list[BoundingBoxSchema],
        assigned: list[int | None],
        matched_tracks: set[int],
    ) -> None:
        """Match detected boxes to active tracks using greedy IoU."""
        for det_idx, det_bbox in enumerate(detected_bboxes):
            best_track_id: int | None = None
            best_iou: float = self.iou_threshold

            for track_id, track in self._active_tracks.items():
                if track_id in matched_tracks:
                    continue
                iou = calculate_iou(det_bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                assigned[det_idx] = best_track_id
                matched_tracks.add(best_track_id)

    def _match_centroid(
        self,
        centroids: list[tuple[float, float]],
        assigned: list[int | None],
        matched_tracks: set[int],
    ) -> None:
        """Match remaining unassigned detections by closest centroid."""
        for det_idx, det_centroid in enumerate(centroids):
            if assigned[det_idx] is not None:
                continue

            best_track_id: int | None = None
            min_dist: float = self.max_centroid_distance

            for track_id, track in self._active_tracks.items():
                if track_id in matched_tracks:
                    continue
                dist = calculate_centroid_distance(det_centroid, track.centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_track_id = track_id

            if best_track_id is not None:
                assigned[det_idx] = best_track_id
                matched_tracks.add(best_track_id)

    def update(
        self, detected_bboxes: list[BoundingBoxSchema]
    ) -> list[tuple[int, BoundingBoxSchema]]:
        """Update tracker with newly detected bounding boxes in current frame."""
        if not detected_bboxes:
            for track in self._active_tracks.values():
                track.frames_since_seen += 1
            self._prune_missing_tracks()
            return []

        centroids = [calculate_centroid(bbox) for bbox in detected_bboxes]
        assigned_track_ids: list[int | None] = [None] * len(detected_bboxes)
        matched_tracks: set[int] = set()

        self._match_iou(detected_bboxes, assigned_track_ids, matched_tracks)
        self._match_centroid(centroids, assigned_track_ids, matched_tracks)

        results: list[tuple[int, BoundingBoxSchema]] = []
        for det_idx, det_bbox in enumerate(detected_bboxes):
            track_id_opt = assigned_track_ids[det_idx]
            centroid = centroids[det_idx]

            if track_id_opt is None:
                new_id = self._next_track_id
                self._next_track_id += 1
                self._active_tracks[new_id] = TrackedFace(
                    track_id=new_id,
                    bbox=det_bbox,
                    centroid=centroid,
                    frames_since_seen=0,
                    total_detections=1,
                )
                results.append((new_id, det_bbox))
            else:
                track = self._active_tracks[track_id_opt]
                track.bbox = det_bbox
                track.centroid = centroid
                track.frames_since_seen = 0
                track.total_detections += 1
                results.append((track_id_opt, det_bbox))

        for track_id, track in self._active_tracks.items():
            if track_id not in matched_tracks:
                track.frames_since_seen += 1

        self._prune_missing_tracks()
        return results

    def reset(self) -> None:
        """Reset all tracked face states and ID counters."""
        self._next_track_id = 1
        self._active_tracks.clear()
