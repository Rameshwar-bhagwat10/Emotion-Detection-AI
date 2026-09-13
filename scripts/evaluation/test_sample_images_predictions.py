"""Test overall accuracy and sample image predictions with the champion model."""

from __future__ import annotations

import io
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import httpx
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.inference.engine import EmotionInferenceEngine
from ml.preprocessing.tensor_pipeline import TensorDataStore

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def create_and_save_sample_images(samples_dir: Path) -> dict[str, list[dict]]:
    """Extract and save diverse test samples for each emotion from the test dataset."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    test_store = TensorDataStore("test")
    
    samples_by_emotion: dict[str, list[dict]] = {c: [] for c in CLASS_NAMES}
    targets = test_store.labels.numpy()
    
    # Select 2 distinct sample images per emotion
    for c_idx, c_name in enumerate(CLASS_NAMES):
        indices = np.where(targets == c_idx)[0]
        # Pick 2 samples from different parts of the test set
        chosen_indices = [indices[10], indices[25]] if len(indices) > 25 else indices[:2]
        
        for s_idx, img_idx in enumerate(chosen_indices):
            # Tensor image is [1, 48, 48] in [0, 1]
            raw_tensor = test_store.images[img_idx]
            img_np = (raw_tensor.squeeze(0).numpy() * 255.0).astype(np.uint8)
            # Upscale cleanly with bicubic interpolation to 224x224 so face detectors easily acquire high quality faces
            img_upscaled = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_CUBIC)
            # Convert to 3-channel BGR for standard image file saving
            img_bgr = cv2.cvtColor(img_upscaled, cv2.COLOR_GRAY2BGR)
            
            file_name = f"sample_{c_name}_{s_idx + 1}.png"
            file_path = samples_dir / file_name
            cv2.imwrite(str(file_path), img_bgr)
            
            samples_by_emotion[c_name].append({
                "path": file_path,
                "label": c_name,
                "label_idx": c_idx,
                "index": int(img_idx),
            })
            
    return samples_by_emotion


def run_sample_verification():
    samples_dir = ROOT_DIR / "data" / "sample_test_images"
    print("=" * 80)
    print("EXTRACTING & VALIDATING SAMPLE TEST IMAGES ACROSS ALL 7 EMOTIONS")
    print("=" * 80)
    
    samples = create_and_save_sample_images(samples_dir)
    print(f"Saved sample test images for all 7 emotions to: {samples_dir}\n")
    
    # Initialize production engine with passthrough face detector for pre-cropped sample faces
    from ml.inference.config import InferencePipelineConfig
    cfg = InferencePipelineConfig()
    cfg.face_detection.detector_type = "passthrough"
    cfg.confidence_threshold = 0.35
    engine = EmotionInferenceEngine(config=cfg)
    
    results = []
    
    for c_name in CLASS_NAMES:
        for s in samples[c_name]:
            img_path = s["path"]
            pred = engine.predict_image(img_path)
            
            # Prediction details
            face_p = pred.faces[0] if pred.faces else None
            if face_p:
                pred_emotion = face_p.emotion
                confidence = face_p.confidence
                is_correct = (pred_emotion.lower() == c_name.lower())
                probs = face_p.probabilities
            else:
                pred_emotion = "no_face"
                confidence = 0.0
                is_correct = False
                probs = {}
                
            results.append({
                "Ground Truth": c_name.upper(),
                "Sample File": img_path.name,
                "Predicted": pred_emotion.upper(),
                "Confidence": f"{confidence*100:.1f}%",
                "Status": "MATCH [OK]" if is_correct else "MISMATCH",
                "Angry Prob": f"{probs.get('angry', 0)*100:.1f}%",
                "Fear Prob": f"{probs.get('fear', 0)*100:.1f}%",
                "Sad Prob": f"{probs.get('sad', 0)*100:.1f}%",
                "Happy Prob": f"{probs.get('happy', 0)*100:.1f}%",
                "Disgust Prob": f"{probs.get('disgust', 0)*100:.1f}%",
            })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    num_correct = sum(1 for r in results if r["Status"].startswith("MATCH"))
    total = len(results)
    print("\n" + "=" * 80)
    print(f"SAMPLE IMAGES ACCURACY: {num_correct} / {total} ({num_correct / total * 100:.1f}%)")
    print("=" * 80)
    
    # Now verify live FastAPI prediction endpoint
    print("\nVERIFYING LIVE FASTAPI API ENDPOINT (/api/v1/predictions):")
    api_url = "http://127.0.0.1:8000/api/v1/predictions"
    api_results = []
    
    for c_name in ["angry", "fear", "sad", "happy", "disgust", "surprise"]:
        s_file = samples_dir / f"sample_{c_name}_1.png"
        try:
            with open(s_file, "rb") as f:
                res = httpx.post(api_url, files={"image": (s_file.name, f, "image/png")}, timeout=10.0)
            if res.status_code == 200:
                data = res.json()
                faces = data.get("faces", [])
                p_em = faces[0].get("emotion", "none") if faces else "no_face"
                p_conf = faces[0].get("confidence", 0) if faces else 0
                api_results.append({
                    "Target Emotion": c_name.upper(),
                    "API Status": f"{res.status_code} OK",
                    "Predicted": p_em.upper(),
                    "Confidence": f"{p_conf*100:.1f}%",
                    "Latency (ms)": f"{data.get('processing_time_ms', 0):.1f}ms",
                    "Match": "PASS" if p_em.lower() == c_name.lower() else "DIFF"
                })
            else:
                api_results.append({
                    "Target Emotion": c_name.upper(),
                    "API Status": f"HTTP {res.status_code}",
                    "Predicted": res.text[:25],
                    "Confidence": "-",
                    "Latency (ms)": "-",
                    "Match": "ERR"
                })
        except Exception as e:
            api_results.append({
                "Target Emotion": c_name.upper(),
                "API Status": f"ERR: {e}",
                "Predicted": "-",
                "Confidence": "-",
                "Latency (ms)": "-",
                "Match": "FAIL"
            })
            
    df_api = pd.DataFrame(api_results)
    print(df_api.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    run_sample_verification()
