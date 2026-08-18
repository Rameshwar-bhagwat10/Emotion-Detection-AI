"""Comprehensive, independent, evidence-based audit suite for Phase 09."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.fer2013.parser import EMOTION_NAMES  # noqa: E402
from ml.inference.config import FaceDetectionConfig, InferencePipelineConfig  # noqa: E402
from ml.inference.engine import EmotionInferenceEngine  # noqa: E402
from ml.inference.face_detector import (  # noqa: E402
    FaceBoundingBox,
    PassThroughFaceDetector,
    YuNetFaceDetector,
    create_face_detector,
)
from ml.inference.image_loader import (  # noqa: E402
    CorruptedImageError,
    ImageNotFoundError,
    InvalidImageDimensionsError,
    load_image,
)
from ml.inference.postprocessing import process_logits  # noqa: E402
from ml.inference.preprocessor import FacePreprocessor, InvalidCropError  # noqa: E402
from ml.inference.schemas import InferenceStatus  # noqa: E402
from ml.models.factory import create_model  # noqa: E402


def run_comprehensive_audit() -> dict:  # noqa: C901
    """Execute complete multi-stage audit for Phase 09."""
    audit_results = {}

    print("=" * 80)
    print("STARTING PHASE 09 INDEPENDENT AUDIT EXECUTION")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. Champion Model Verification
    # --------------------------------------------------------------------------
    meta_path = PROJECT_ROOT / "artifacts" / "optimized" / "champion" / "metadata.json"
    weights_path = PROJECT_ROOT / "artifacts" / "optimized" / "champion" / "model.pt"
    onnx_path = PROJECT_ROOT / "artifacts" / "optimized" / "champion" / "model.onnx"

    assert meta_path.exists(), f"Metadata missing at {meta_path}"
    assert weights_path.exists(), f"Weights missing at {weights_path}"
    assert onnx_path.exists(), f"ONNX missing at {onnx_path}"

    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    audit_results["champion_model_name"] = metadata.get("model_name")
    audit_results["champion_base_model"] = metadata.get("base_model")
    audit_results["champion_optimization"] = metadata.get("optimization_technique")
    audit_results["weights_size_bytes"] = weights_path.stat().st_size
    audit_results["onnx_size_bytes"] = onnx_path.stat().st_size

    # Load model and verify architecture & state dict
    model = create_model(metadata["base_model"])
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    # Forward pass test
    dummy_input = torch.randn(2, 1, 48, 48, dtype=torch.float32)
    with torch.inference_mode():
        dummy_out = model(dummy_input)

    assert dummy_out.shape == (2, 7), f"Expected shape (2, 7), got {dummy_out.shape}"
    audit_results["model_forward_shape"] = list(dummy_out.shape)
    audit_results["model_loading_verified"] = True
    print("[PASS] 1. Champion Model Verification & Integrity")

    # --------------------------------------------------------------------------
    # 2. Hardcoded Path Audit
    # --------------------------------------------------------------------------
    inf_dir = PROJECT_ROOT / "ml" / "inference"
    hardcoded_found = []
    for py_file in inf_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if "C:\\Users\\" in text or "D:\\" in text or "/home/" in text:
            hardcoded_found.append(py_file.name)

    audit_results["hardcoded_paths_found"] = hardcoded_found
    assert len(hardcoded_found) == 0, f"Hardcoded paths found in {hardcoded_found}"
    print("[PASS] 2. Hardcoded Path Audit (0 hardcoded paths found)")

    # --------------------------------------------------------------------------
    # 3. Model Immutability Test
    # --------------------------------------------------------------------------
    engine = EmotionInferenceEngine(auto_load=True)
    assert engine.model is not None
    param_hashes_before = [torch.sum(p).item() for p in engine.model.parameters()]

    # Run 25 predictions
    synth_img = np.full((100, 100, 3), 150, dtype=np.uint8)
    for _ in range(25):
        _ = engine.predict_image(synth_img)

    param_hashes_after = [torch.sum(p).item() for p in engine.model.parameters()]
    assert param_hashes_before == param_hashes_after, "Model parameters mutated during inference!"
    audit_results["immutability_verified"] = True
    print("[PASS] 3. Model Immutability Test (parameters 100% invariant across 25 inferences)")

    # --------------------------------------------------------------------------
    # 4. Image Validation & Loading Test
    # --------------------------------------------------------------------------
    # Valid NumPy
    np_img = np.zeros((80, 80, 3), dtype=np.uint8)
    loaded_np, fmt_np = load_image(np_img)
    assert loaded_np.shape == (80, 80, 3)

    # Valid PIL
    pil_img = Image.new("RGB", (60, 60), color="blue")
    loaded_pil, fmt_pil = load_image(pil_img)
    assert loaded_pil.shape == (60, 60, 3)

    # Invalid path
    try:
        load_image("invalid_nonexistent_path_xyz.jpg")
        inv_path_handled = False
    except ImageNotFoundError:
        inv_path_handled = True

    # Corrupt bytes
    try:
        load_image(b"broken corrupted bytes")
        corrupt_handled = False
    except CorruptedImageError:
        corrupt_handled = True

    # Oversized
    try:
        load_image(np.zeros((5000, 5000, 3), dtype=np.uint8), max_dimension=4096)
        oversized_handled = False
    except InvalidImageDimensionsError:
        oversized_handled = True

    assert inv_path_handled and corrupt_handled and oversized_handled
    audit_results["image_validation_verified"] = True
    print("[PASS] 4. Image Ingestion & Validation Test")

    # --------------------------------------------------------------------------
    # 5. Face Detection & Bounding Box Operations
    # --------------------------------------------------------------------------
    bbox = FaceBoundingBox(x=20, y=30, width=50, height=60)
    assert bbox.area == 3000
    assert bbox.x1 == 20 and bbox.y1 == 30 and bbox.x2 == 70 and bbox.y2 == 90

    # 15% padding
    padded = bbox.pad_and_clip(padding_fraction=0.15, img_width=100, img_height=100)
    assert padded.x < bbox.x and padded.y < bbox.y
    assert padded.width > bbox.width and padded.height > bbox.height

    # Edge clipping
    edge_box = FaceBoundingBox(x=0, y=0, width=50, height=50)
    padded_edge = edge_box.pad_and_clip(padding_fraction=0.50, img_width=50, img_height=50)
    assert padded_edge.x == 0 and padded_edge.y == 0
    assert padded_edge.x2 <= 50 and padded_edge.y2 <= 50

    # Test YuNet & PassThrough detectors
    yunet = YuNetFaceDetector()
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    yunet_blank_res = yunet.detect(blank_img)
    assert len(yunet_blank_res) == 0

    pt_detector = PassThroughFaceDetector()
    pt_res = pt_detector.detect(blank_img)
    assert len(pt_res) == 1 and pt_res[0].confidence == 1.0

    audit_results["face_detection_verified"] = True
    print("[PASS] 5. Face Detection & Bounding Box Padding/Clipping Test")

    # --------------------------------------------------------------------------
    # 6. Preprocessing Consistency Audit
    # --------------------------------------------------------------------------
    preprocessor = FacePreprocessor()
    crop_rgb = np.full((60, 60, 3), 128, dtype=np.uint8)
    tensor = preprocessor.preprocess_single_crop(crop_rgb)

    assert tensor.shape == (1, 48, 48)
    assert tensor.dtype == torch.float32
    # Verify training split normalization exact calculation
    expected_norm = (128.0 / 255.0 - 0.507743) / 0.255009
    assert abs(tensor[0, 0, 0].item() - expected_norm) < 1e-4

    # Test batch tensor construction
    crops = [crop_rgb, crop_rgb, crop_rgb]
    batch_tensor = preprocessor.preprocess_batch(crops, device="cpu")
    assert batch_tensor.shape == (3, 1, 48, 48)

    # Test invalid crop
    try:
        preprocessor.crop_face(
            np.zeros((100, 100, 3), dtype=np.uint8), FaceBoundingBox(x=0, y=0, width=0, height=0)
        )
        invalid_crop_handled = False
    except InvalidCropError:
        invalid_crop_handled = True

    assert invalid_crop_handled
    audit_results["preprocessing_consistency_verified"] = True
    print("[PASS] 6. Preprocessing Consistency with Training Split Test")

    # --------------------------------------------------------------------------
    # 7. Postprocessing, Softmax & Confidence Gating Audit
    # --------------------------------------------------------------------------
    fake_logits = torch.tensor(
        [
            [-2.0, -3.0, -1.0, 6.0, -2.0, -1.0, -2.0],  # High confidence 'happy' (idx 3)
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Uniform / Low confidence (all equal ~14%)
        ],
        dtype=torch.float32,
    )
    detections = [
        create_face_detector(detector_type="passthrough").detect(
            np.zeros((50, 50, 3), dtype=np.uint8)
        )[0],
        create_face_detector(detector_type="passthrough").detect(
            np.zeros((50, 50, 3), dtype=np.uint8)
        )[0],
    ]

    preds = process_logits(
        logits=fake_logits,
        detections=detections,
        classes=list(EMOTION_NAMES),
        confidence_threshold=0.40,
    )

    assert len(preds) == 2
    # Sample 1: Happy, High confidence, not uncertain
    assert preds[0].emotion == "happy"
    assert preds[0].confidence > 0.90
    assert preds[0].is_uncertain is False
    assert abs(sum(preds[0].probabilities.values()) - 1.0) < 1e-4

    # Sample 2: Low confidence (< 0.40), marked uncertain, all 7 probabilities intact
    assert preds[1].emotion == "uncertain"
    assert preds[1].confidence < 0.40
    assert preds[1].is_uncertain is True
    assert len(preds[1].probabilities) == 7

    audit_results["postprocessing_verified"] = True
    print("[PASS] 7. Postprocessing, Softmax & Confidence Decision Gate Test")

    # --------------------------------------------------------------------------
    # 8. End-to-End Pipeline Execution (Single, Multi, Zero Face)
    # --------------------------------------------------------------------------
    # A. Zero face
    blank_image = np.zeros((200, 200, 3), dtype=np.uint8)
    res_zero = engine.predict_image(blank_image)
    assert res_zero.status == InferenceStatus.NO_FACE_DETECTED
    assert res_zero.faces_detected == 0
    assert len(res_zero.faces) == 0

    # B. Single face crop (using passthrough)
    pt_engine = EmotionInferenceEngine(
        config=InferencePipelineConfig(
            face_detection=FaceDetectionConfig(detector_type="passthrough")
        ),
        auto_load=True,
    )
    sample_face = np.full((120, 120, 3), 160, dtype=np.uint8)
    res_single = pt_engine.predict_image(sample_face)
    assert res_single.status == InferenceStatus.SUCCESS
    assert res_single.faces_detected == 1
    assert len(res_single.faces) == 1
    assert (
        res_single.faces[0].emotion in EMOTION_NAMES or res_single.faces[0].emotion == "uncertain"
    )
    assert res_single.timing is not None and res_single.timing.total_ms > 0

    # C. Prediction Determinism
    res_single_repeat = pt_engine.predict_image(sample_face)
    assert res_single.faces[0].confidence == res_single_repeat.faces[0].confidence
    for k in res_single.faces[0].probabilities:
        assert (
            abs(res_single.faces[0].probabilities[k] - res_single_repeat.faces[0].probabilities[k])
            < 1e-6
        )

    # D. Output Schema JSON Serialization
    dict_repr = res_single.to_dict()
    json_str = json.dumps(dict_repr)
    assert len(json_str) > 50
    assert "status" in dict_repr and dict_repr["status"] == "SUCCESS"

    audit_results["e2e_pipeline_verified"] = True
    print("[PASS] 8. End-to-End Prediction Pipeline & JSON Serialization Test")

    # --------------------------------------------------------------------------
    # 9. ONNX Export Consistency Test
    # --------------------------------------------------------------------------
    import onnxruntime as ort

    onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {onnx_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = onnx_session.run(None, ort_inputs)
    onnx_logits = ort_outs[0]

    py_logits = dummy_out.numpy()
    max_diff = float(np.max(np.abs(py_logits - onnx_logits)))
    assert max_diff < 1e-4, f"PyTorch and ONNX logits differ significantly: max diff = {max_diff}"
    audit_results["onnx_max_absolute_error"] = max_diff
    audit_results["onnx_fidelity_verified"] = True
    print(f"[PASS] 9. ONNX Export Numerical Consistency Test (max error = {max_diff:.2e})")

    # --------------------------------------------------------------------------
    # 10. Scope & Safety Audit
    # --------------------------------------------------------------------------
    scope_violations = []
    for py_path in (PROJECT_ROOT / "ml" / "inference").glob("**/*.py"):
        content = py_path.read_text(encoding="utf-8")
        if "VideoCapture(" in content:
            scope_violations.append(f"{py_path.name}: VideoCapture found")
        if "FastAPI(" in content or "Flask(" in content:
            scope_violations.append(f"{py_path.name}: Web server found")

    assert len(scope_violations) == 0, f"Scope violations detected: {scope_violations}"
    audit_results["scope_clean"] = True
    print("[PASS] 10. Scope Compliance Audit (0 prohibited modules in Phase 09)")

    print("=" * 80)
    print("ALL 10 DEEP AUDIT SUITES EXECUTED AND PASSED")
    print("=" * 80)
    return audit_results


if __name__ == "__main__":
    results = run_comprehensive_audit()
    out_file = PROJECT_ROOT / "scratch" / "audit_phase09_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Audit results written to {out_file}")
