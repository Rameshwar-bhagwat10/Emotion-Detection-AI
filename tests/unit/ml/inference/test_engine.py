"""Unit tests for EmotionInferenceEngine orchestration."""

from __future__ import annotations

import numpy as np

from ml.inference.config import FaceDetectionConfig, InferencePipelineConfig
from ml.inference.engine import EmotionInferenceEngine
from ml.inference.schemas import InferenceStatus


def test_engine_no_face_detected():
    """Verify engine returns NO_FACE_DETECTED for blank image."""
    config = InferencePipelineConfig(face_detection=FaceDetectionConfig(detector_type="haar"))
    engine = EmotionInferenceEngine(config=config, auto_load=True)

    blank_img = np.full((300, 300, 3), 128, dtype=np.uint8)
    result = engine.predict_image(blank_img)

    assert result.status == InferenceStatus.NO_FACE_DETECTED
    assert result.faces_detected == 0
    assert len(result.faces) == 0
    assert result.image is not None
    assert result.image.width == 300
    assert result.image.height == 300
    assert result.timing is not None
    assert result.timing.face_detection_ms >= 0


def test_engine_passthrough_face_prediction():
    """Verify complete end-to-end prediction with PassThrough detector."""
    config = InferencePipelineConfig(
        face_detection=FaceDetectionConfig(detector_type="passthrough")
    )
    engine = EmotionInferenceEngine(config=config, auto_load=True)

    # 48x48 sample face image
    sample_face = np.full((48, 48, 3), 150, dtype=np.uint8)
    result = engine.predict_image(sample_face)

    assert result.status == InferenceStatus.SUCCESS
    assert result.faces_detected == 1
    assert len(result.faces) == 1

    face = result.faces[0]
    assert face.face_id == 1
    assert face.emotion in config.classes or face.emotion == "uncertain"
    assert 0.0 <= face.confidence <= 1.0
    assert len(face.probabilities) == 7
    assert abs(sum(face.probabilities.values()) - 1.0) < 1e-3

    # Check timing
    assert result.timing is not None
    assert result.timing.total_ms > 0
    assert result.timing.inference_ms >= 0

    # Check serialization
    res_dict = result.to_dict()
    assert res_dict["status"] == "SUCCESS"
    assert "faces" in res_dict
    assert "timing" in res_dict


def test_engine_invalid_input_handling():
    """Verify graceful handling of invalid file path."""
    engine = EmotionInferenceEngine(auto_load=True)
    result = engine.predict_image("non_existent_file_path_xyz.jpg")

    assert result.status == InferenceStatus.INVALID_IMAGE
    assert result.error_message is not None
    assert "Image not found" in result.error_message


def test_engine_predict_batch():
    """Verify batch prediction across multiple image sources."""
    config = InferencePipelineConfig(
        face_detection=FaceDetectionConfig(detector_type="passthrough")
    )
    engine = EmotionInferenceEngine(config=config, auto_load=True)

    images = [
        np.full((48, 48, 3), 100, dtype=np.uint8),
        np.full((48, 48, 3), 200, dtype=np.uint8),
    ]

    results = engine.predict_batch(images)
    assert len(results) == 2
    assert all(r.status == InferenceStatus.SUCCESS for r in results)
