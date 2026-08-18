"""Integration test suite for the Phase 09 Inference Engine and Prediction Pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from ml.inference.config import FaceDetectionConfig, InferencePipelineConfig
from ml.inference.engine import EmotionInferenceEngine
from ml.inference.face_detector import FaceBoundingBox
from ml.inference.predictor import EmotionPredictor
from ml.inference.schemas import InferenceStatus


@pytest.fixture
def inference_engine() -> EmotionInferenceEngine:
    """Fixture providing a warmed-up inference engine with PassThrough detector for deterministic tests."""
    cfg = InferencePipelineConfig(
        face_detection=FaceDetectionConfig(detector_type="passthrough", face_padding=0.15),
        confidence_threshold=0.40,
        warmup_iterations=2,
    )
    return EmotionInferenceEngine(config=cfg, auto_load=True)


def test_integration_single_face_inference_e2e(inference_engine: EmotionInferenceEngine):
    """Scenario 1: End-to-end single face inference with probability distribution."""
    # Synthetic face image
    face_img = np.full((120, 120, 3), 160, dtype=np.uint8)

    result = inference_engine.predict_image(face_img)

    assert result.status == InferenceStatus.SUCCESS
    assert result.faces_detected == 1
    assert len(result.faces) == 1

    face = result.faces[0]
    assert face.face_id == 1
    assert face.emotion in inference_engine.config.classes or face.emotion == "uncertain"
    assert 0.0 <= face.confidence <= 1.0
    assert len(face.probabilities) == 7
    assert abs(sum(face.probabilities.values()) - 1.0) < 1e-3

    # Verify timing breakdown presence
    assert result.timing is not None
    assert result.timing.image_loading_ms >= 0
    assert result.timing.face_detection_ms >= 0
    assert result.timing.preprocessing_ms >= 0
    assert result.timing.inference_ms >= 0
    assert result.timing.postprocessing_ms >= 0
    assert result.timing.total_ms > 0

    # Verify ModelInfo
    assert result.model_info is not None
    assert result.model_info.model_name == "champion-pruning-30"


def test_integration_zero_face_detection():
    """Scenario 2: Zero faces in blank image produces NO_FACE_DETECTED."""
    cfg = InferencePipelineConfig(face_detection=FaceDetectionConfig(detector_type="haar"))
    engine = EmotionInferenceEngine(config=cfg, auto_load=True)

    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    res = engine.predict_image(blank)

    assert res.status == InferenceStatus.NO_FACE_DETECTED
    assert res.faces_detected == 0
    assert len(res.faces) == 0
    assert res.timing is not None
    assert res.timing.face_detection_ms >= 0


def test_integration_invalid_and_corrupted_inputs(
    inference_engine: EmotionInferenceEngine, tmp_path: Path
):
    """Scenario 3: Corrupted bytes and missing files handled gracefully without crashing."""
    # Non-existent file
    res_missing = inference_engine.predict_image("non_existent_image_12345.jpg")
    assert res_missing.status == InferenceStatus.INVALID_IMAGE
    assert res_missing.error_message is not None

    # Corrupt file
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"garbage content")
    res_corrupt = inference_engine.predict_image(corrupt_file)
    assert res_corrupt.status == InferenceStatus.INVALID_IMAGE
    assert res_corrupt.error_message is not None


def test_integration_bounding_box_padding_and_clipping():
    """Scenario 4: Bounding-box padding expanding beyond borders clips safely."""
    bbox = FaceBoundingBox(x=10, y=10, width=50, height=50)
    # 50% padding on a 60x60 image will push coords below 0 and above 60
    padded = bbox.pad_and_clip(padding_fraction=0.50, img_width=60, img_height=60)
    assert padded.x == 0
    assert padded.y == 0
    assert padded.x + padded.width <= 60
    assert padded.y + padded.height <= 60


def test_integration_confidence_gating():
    """Scenario 5: High threshold forces 'uncertain' status while preserving probabilities."""
    # Set impossible threshold of 0.999
    cfg_strict = InferencePipelineConfig(
        face_detection=FaceDetectionConfig(detector_type="passthrough"),
        confidence_threshold=0.999,
    )
    engine_strict = EmotionInferenceEngine(config=cfg_strict, auto_load=True)

    face_img = np.full((48, 48, 3), 120, dtype=np.uint8)
    res = engine_strict.predict_image(face_img)

    assert res.status == InferenceStatus.SUCCESS
    assert len(res.faces) == 1
    assert res.faces[0].is_uncertain is True
    assert res.faces[0].emotion == "uncertain"
    # Probabilities must still be preserved
    assert len(res.faces[0].probabilities) == 7


def test_integration_model_immutability(inference_engine: EmotionInferenceEngine):
    """Scenario 6: Model parameters remain 100% identical after multiple predictions."""
    assert inference_engine.model is not None
    weights_before = [p.detach().clone() for p in inference_engine.model.parameters()]

    # Execute 10 inference calls
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    for _ in range(10):
        _ = inference_engine.predict_image(img)

    # Compare parameters
    for p_before, p_after in zip(weights_before, inference_engine.model.parameters(), strict=True):
        assert torch.equal(p_before, p_after)


def test_integration_prediction_determinism(inference_engine: EmotionInferenceEngine):
    """Scenario 7: Repeated inference on exact same image yields identical probability values."""
    img = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)

    res1 = inference_engine.predict_image(img)
    res2 = inference_engine.predict_image(img)

    assert res1.faces[0].emotion == res2.faces[0].emotion
    assert res1.faces[0].confidence == res2.faces[0].confidence
    for k in res1.faces[0].probabilities:
        assert abs(res1.faces[0].probabilities[k] - res2.faces[0].probabilities[k]) < 1e-6


def test_integration_input_format_diversity(
    inference_engine: EmotionInferenceEngine, tmp_path: Path
):
    """Scenario 8: Support for PIL Image, raw bytes, file path, and NumPy array."""
    sample_np = np.full((50, 50, 3), 130, dtype=np.uint8)

    # 1. NumPy
    res_np = inference_engine.predict_image(sample_np)
    assert res_np.status == InferenceStatus.SUCCESS

    # 2. PIL Image
    pil_img = Image.fromarray(sample_np)
    res_pil = inference_engine.predict_image(pil_img)
    assert res_pil.status == InferenceStatus.SUCCESS

    # 3. File Path
    file_path = tmp_path / "test.png"
    cv2.imwrite(str(file_path), cv2.cvtColor(sample_np, cv2.COLOR_RGB2BGR))
    res_file = inference_engine.predict_image(file_path)
    assert res_file.status == InferenceStatus.SUCCESS

    # 4. Bytes
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    res_bytes = inference_engine.predict_image(bytes_data)
    assert res_bytes.status == InferenceStatus.SUCCESS


def test_integration_emotion_predictor_convenience():
    """Scenario 9: Test high-level EmotionPredictor convenience wrapper."""
    cfg = InferencePipelineConfig(face_detection=FaceDetectionConfig(detector_type="passthrough"))
    predictor = EmotionPredictor(config=cfg)

    img = np.full((48, 48, 3), 150, dtype=np.uint8)
    res = predictor.predict(img)
    assert res.status == InferenceStatus.SUCCESS

    face = predictor.predict_single_face_crop(img)
    assert face.face_id == 1
    assert len(face.probabilities) == 7
