"""Computer vision face detection module."""

from __future__ import annotations

from ml.inference.face_detector import (
    BaseFaceDetector,
    FaceBoundingBox,
    FaceDetection,
    HaarCascadeFaceDetector,
    PassThroughFaceDetector,
    YuNetFaceDetector,
    create_face_detector,
)

__all__ = [
    "FaceBoundingBox",
    "FaceDetection",
    "BaseFaceDetector",
    "HaarCascadeFaceDetector",
    "YuNetFaceDetector",
    "PassThroughFaceDetector",
    "create_face_detector",
]
