"""Unit tests for face detection backends and bounding box operations."""

from __future__ import annotations

import numpy as np
import pytest

from ml.inference.face_detector import (
    FaceBoundingBox,
    HaarCascadeFaceDetector,
    PassThroughFaceDetector,
    YuNetFaceDetector,
    create_face_detector,
)


def test_bounding_box_properties_and_area():
    """Verify coordinate accessors and area calculations."""
    bbox = FaceBoundingBox(x=10, y=20, width=50, height=60)
    assert bbox.x1 == 10
    assert bbox.y1 == 20
    assert bbox.x2 == 60
    assert bbox.y2 == 80
    assert bbox.area == 3000
    assert bbox.to_tuple() == (10, 20, 50, 60)


def test_bounding_box_pad_and_clip():
    """Verify symmetric expansion and boundary clipping."""
    bbox = FaceBoundingBox(x=50, y=50, width=100, height=100)
    # 20% padding expands by 10px on each side
    padded = bbox.pad_and_clip(padding_fraction=0.20, img_width=300, img_height=300)
    assert padded.x == 40
    assert padded.y == 40
    assert padded.width == 120
    assert padded.height == 120

    # Boundary clipping test (exceeding top-left)
    edge_box = FaceBoundingBox(x=5, y=5, width=50, height=50)
    padded_edge = edge_box.pad_and_clip(padding_fraction=0.50, img_width=200, img_height=200)
    assert padded_edge.x == 0
    assert padded_edge.y == 0

    # Boundary clipping test (exceeding bottom-right)
    bottom_box = FaceBoundingBox(x=180, y=180, width=50, height=50)
    padded_bottom = bottom_box.pad_and_clip(padding_fraction=0.50, img_width=200, img_height=200)
    assert padded_bottom.x2 <= 200
    assert padded_bottom.y2 <= 200


def test_yunet_detector_blank_image():
    """Verify YuNet detector handles blank images without false positives."""
    detector = YuNetFaceDetector()
    blank_img = np.full((300, 300, 3), 128, dtype=np.uint8)
    detections = detector.detect(blank_img)
    assert isinstance(detections, list)
    assert len(detections) == 0


def test_haar_detector_instantiation_and_empty_image():
    """Verify Haar Cascade detector handles blank images gracefully without faces."""
    detector = HaarCascadeFaceDetector()
    blank_img = np.full((300, 300, 3), 128, dtype=np.uint8)
    detections = detector.detect(blank_img)
    assert isinstance(detections, list)
    assert len(detections) == 0


def test_passthrough_detector():
    """Verify PassThroughFaceDetector returns entire image as face 1."""
    detector = PassThroughFaceDetector()
    img = np.zeros((150, 200, 3), dtype=np.uint8)
    detections = detector.detect(img)
    assert len(detections) == 1
    assert detections[0].face_id == 1
    assert detections[0].bbox.width == 200
    assert detections[0].bbox.height == 150
    assert detections[0].confidence == 1.0


def test_create_face_detector_factory():
    """Verify factory returns appropriate detector instances."""
    det_yunet = create_face_detector(detector_type="yunet")
    assert isinstance(det_yunet, YuNetFaceDetector)

    det_haar = create_face_detector(detector_type="haar")
    assert isinstance(det_haar, HaarCascadeFaceDetector)

    det_pt = create_face_detector(detector_type="passthrough")
    assert isinstance(det_pt, PassThroughFaceDetector)

    with pytest.raises(ValueError):
        create_face_detector(detector_type="unknown_type")
