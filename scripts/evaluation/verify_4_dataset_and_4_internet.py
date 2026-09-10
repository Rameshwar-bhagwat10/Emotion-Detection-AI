"""Comprehensive evaluation on 4 random test set images + 4 real-world internet images using YuNet face detection."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import urllib.request
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

torch.set_num_threads(14)

from ml.models.factory import create_model
from ml.inference.face_detector import create_face_detector
from ml.preprocessing.tensor_pipeline import TensorDataStore, IMAGENET_MEAN, IMAGENET_STD

OUTPUT_DIR = ROOT_DIR / "reports" / "random_eval_run"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# 4 High-Resolution Real-World Portrait Photos from Public Unsplash CDN with distinct emotions
INTERNET_IMAGE_URLS = [
    {
        "name": "internet_1_happy",
        "expected_expression": "HAPPY",
        "url": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=600&auto=format&fit=crop&q=80",
        "description": "Joyful smiling woman portrait with authentic beaming smile",
    },
    {
        "name": "internet_2_surprise",
        "expected_expression": "SURPRISE",
        "url": "https://images.unsplash.com/photo-1578632767115-351597cf2477?w=600&auto=format&fit=crop&q=80",
        "description": "Person with wide open eyes and dropped mouth expressing shock/surprise",
    },
    {
        "name": "internet_3_sad",
        "expected_expression": "SAD",
        "url": "https://images.unsplash.com/photo-1509967419530-da38b4704bc6?w=600&auto=format&fit=crop&q=80",
        "description": "Somber melancholic portrait with downcast eyes and sorrowful expression",
    },
    {
        "name": "internet_4_angry",
        "expected_expression": "ANGRY",
        "url": "https://images.unsplash.com/photo-1595152772835-219674b2a8a6?w=600&auto=format&fit=crop&q=80",
        "description": "Intense furrowed-brow glare with fierce scowl (Angry)",
    },
]


def load_model():
    device = torch.device("cpu")
    model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    weights_path = ROOT_DIR / "models" / "v2_optimized" / "model.pt"
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device)
    model.eval()
    return model, device


def predict_48x48_gray(model, device, gray_48: np.ndarray):
    t_img = torch.from_numpy(gray_48).unsqueeze(0).unsqueeze(0).float() / 255.0  # [1, 1, 48, 48]
    t_112 = F.interpolate(t_img, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    t_norm = (t_112 - IMAGENET_MEAN) / IMAGENET_STD

    with torch.inference_mode():
        logits = model(t_norm)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]
        conf = float(probs[pred_idx])

    prob_dict = {CLASS_NAMES[k]: round(float(probs[k]) * 100, 2) for k in range(7)}
    return pred_name, conf, prob_dict


def run_evaluation():
    model, device = load_model()
    detector = create_face_detector(detector_type="yunet")

    # =========================================================================
    # PART 1: 4 RANDOM DATASET TEST IMAGES
    # =========================================================================
    print("=" * 75)
    print("PART 1: EVALUATION ON 4 RANDOM TEST SET IMAGES")
    print("=" * 75)

    test_store = TensorDataStore("test")
    raw_imgs = (test_store.images.squeeze(1).numpy() * 255.0).astype(np.uint8)
    labels = test_store.labels.numpy()

    # Pick 4 random images across diverse categories: Happy, Sad, Surprise, Angry
    np.random.seed(2026)
    target_classes = [3, 4, 5, 0]  # Happy, Sad, Surprise, Angry
    chosen_indices = [int(np.random.choice(np.where(labels == c)[0])) for c in target_classes]

    dataset_results = []
    ds_correct = 0

    for i, idx in enumerate(chosen_indices, 1):
        img_arr = raw_imgs[idx]
        true_name = CLASS_NAMES[int(labels[idx])].upper()

        save_path = OUTPUT_DIR / f"dataset_random_{i}_{true_name.lower()}.png"
        Image.fromarray(img_arr).resize((112, 112), Image.Resampling.NEAREST).save(save_path)

        pred_name, conf, prob_dict = predict_48x48_gray(model, device, img_arr)
        is_correct = (pred_name.upper() == true_name)
        if is_correct:
            ds_correct += 1

        top_probs = ", ".join([f"{k}: {v}%" for k, v in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]])

        dataset_results.append({
            "Sample": f"Dataset #{i}",
            "True Emotion": true_name,
            "Predicted Emotion": pred_name.upper(),
            "Confidence": f"{conf*100:.1f}%",
            "Result": "CORRECT" if is_correct else "INCORRECT",
            "Top Probabilities": top_probs,
            "Full Probabilities": prob_dict,
            "Saved Image": str(save_path),
        })

    ds_df = pd.DataFrame(dataset_results)
    print(ds_df[["Sample", "True Emotion", "Predicted Emotion", "Confidence", "Result", "Top Probabilities"]].to_string(index=False))
    print(f"\nDataset 4-Sample Accuracy: {ds_correct}/4 ({ds_correct/4*100:.1f}%)\n")

    # =========================================================================
    # PART 2: 4 REAL-WORLD INTERNET IMAGES (YUNET DETECTION + PREDICTION)
    # =========================================================================
    print("=" * 75)
    print("PART 2: EVALUATION ON 4 REAL-WORLD INTERNET IMAGES")
    print("=" * 75)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    internet_results = []
    net_correct = 0

    for item in INTERNET_IMAGE_URLS:
        print(f"Downloading & detecting face: {item['name']} ({item['expected_expression']})...")
        try:
            req = urllib.request.Request(item["url"], headers=headers)
            raw_bytes = urllib.request.urlopen(req, timeout=15).read()
            pil_orig = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            orig_save_path = OUTPUT_DIR / f"{item['name']}_orig.jpg"
            pil_orig.save(orig_save_path)

            np_rgb = np.array(pil_orig)
            bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

            # YuNet face detection
            detections = detector.detect(bgr)
            if detections:
                best_det = max(detections, key=lambda d: d.bbox.area)
                bbox = best_det.bbox
                # Add 10% padding
                img_h, img_w, _ = bgr.shape
                pad_w = int(bbox.width * 0.10)
                pad_h = int(bbox.height * 0.10)
                x0 = max(0, bbox.x - pad_w)
                y0 = max(0, bbox.y - pad_h)
                x1 = min(img_w, bbox.x + bbox.width + pad_w)
                y1 = min(img_h, bbox.y + bbox.height + pad_h)
                cropped_rgb = np_rgb[y0:y1, x0:x1]
            else:
                # Fallback center crop
                h, w, _ = np_rgb.shape
                min_dim = min(h, w)
                cy, cx = int(h * 0.45), int(w * 0.50)
                half = int(min_dim * 0.38)
                cropped_rgb = np_rgb[max(0, cy-half):min(h, cy+half), max(0, cx-half):min(w, cx+half)]

            # Convert to 48x48 grayscale
            face_pil = Image.fromarray(cropped_rgb).convert("L")
            face_48 = np.array(face_pil.resize((48, 48), Image.Resampling.BILINEAR), dtype=np.uint8)

            crop_save_path = OUTPUT_DIR / f"{item['name']}_face_crop.png"
            Image.fromarray(face_48).resize((112, 112), Image.Resampling.NEAREST).save(crop_save_path)

            pred_name, conf, prob_dict = predict_48x48_gray(model, device, face_48)
            is_correct = (pred_name.upper() == item["expected_expression"])
            if is_correct:
                net_correct += 1

            top_probs = ", ".join([f"{k}: {v}%" for k, v in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]])

            internet_results.append({
                "Image Name": item["name"],
                "Expected Emotion": item["expected_expression"],
                "Predicted Emotion": pred_name.upper(),
                "Confidence": f"{conf*100:.1f}%",
                "Result": "CORRECT" if is_correct else "INCORRECT",
                "Top Probabilities": top_probs,
                "Full Probabilities": prob_dict,
                "Description": item["description"],
                "Saved Crop": str(crop_save_path),
            })
        except Exception as e:
            print(f"Error processing {item['name']}: {e}")

    net_df = pd.DataFrame(internet_results)
    print("\n" + net_df[["Image Name", "Expected Emotion", "Predicted Emotion", "Confidence", "Result", "Top Probabilities"]].to_string(index=False))
    print(f"\nInternet 4-Sample Accuracy: {net_correct}/{len(internet_results)} ({net_correct/max(len(internet_results), 1)*100:.1f}%)")

    # Save detailed JSON summary
    summary = {
        "dataset_evaluation": {
            "samples": dataset_results,
            "accuracy": f"{ds_correct}/4 ({ds_correct/4*100:.1f}%)",
        },
        "internet_evaluation": {
            "samples": internet_results,
            "accuracy": f"{net_correct}/{len(internet_results)} ({net_correct/max(len(internet_results), 1)*100:.1f}%)",
        },
    }

    with open(OUTPUT_DIR / "verification_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_evaluation()
