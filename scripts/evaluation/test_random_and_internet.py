"""Evaluate Optimized Emotion Detection Model on 4 Random Dataset Images + 4 Real-World Internet Images."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import urllib.request
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
from ml.preprocessing.tensor_pipeline import TensorDataStore, IMAGENET_MEAN, IMAGENET_STD

OUTPUT_DIR = ROOT_DIR / "reports" / "random_and_internet_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# 4 High-Resolution Real-World Portrait Photos from Public Unsplash CDN
INTERNET_IMAGE_URLS = [
    {
        "name": "internet_1_happy_smile",
        "expected_expression": "HAPPY",
        "url": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=500&auto=format&fit=crop&q=80",
        "description": "Smiling female portrait with authentic joyful smile",
    },
    {
        "name": "internet_2_surprise",
        "expected_expression": "SURPRISE",
        "url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=500&auto=format&fit=crop&q=80",
        "description": "Female portrait with wide eyes and elevated eyebrows (Surprise)",
    },
    {
        "name": "internet_3_sad",
        "expected_expression": "SAD",
        "url": "https://images.unsplash.com/photo-1509967419530-da38b4704bc6?w=500&auto=format&fit=crop&q=80",
        "description": "Somber portrait with melancholic downcast facial expression (Sad)",
    },
    {
        "name": "internet_4_neutral",
        "expected_expression": "NEUTRAL",
        "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=500&auto=format&fit=crop&q=80",
        "description": "Male portrait with calm resting neutral facial expression (Neutral)",
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


def crop_portrait_face_pil(pil_img: Image.Image) -> np.ndarray:
    """Center square crop focused on face region for portrait photography."""
    w, h = pil_img.size
    min_dim = min(w, h)
    # Focus slightly above center (where eyes/face sit in portraits)
    cy = int(h * 0.45)
    cx = int(w * 0.50)
    
    half_size = int(min_dim * 0.38)
    x0 = max(0, cx - half_size)
    y0 = max(0, cy - half_size)
    x1 = min(w, cx + half_size)
    y1 = min(h, cy + half_size)
    
    cropped = pil_img.crop((x0, y0, x1, y1)).convert("L")
    resized = cropped.resize((48, 48), Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.uint8)


def predict_img_arr(model, device, arr_48: np.ndarray):
    t_img = torch.from_numpy(arr_48).unsqueeze(0).unsqueeze(0).float() / 255.0  # [1, 1, 48, 48]
    t_112 = F.interpolate(t_img, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    t_norm = (t_112 - IMAGENET_MEAN) / IMAGENET_STD

    with torch.inference_mode():
        logits = model(t_norm)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]
        conf = float(probs[pred_idx])

    prob_dict = {CLASS_NAMES[k]: round(float(probs[k]) * 100, 1) for k in range(7)}
    return pred_name, conf, prob_dict


def run_evaluation():
    model, device = load_model()
    print("=" * 70)
    print("1. EVALUATION ON 4 RANDOM DATASET TEST IMAGES")
    print("=" * 70)

    test_store = TensorDataStore("test")
    raw_imgs = (test_store.images.squeeze(1).numpy() * 255.0).astype(np.uint8)
    labels = test_store.labels.numpy()

    # 4 distinct random emotion classes
    np.random.seed(42)
    target_classes = [3, 0, 4, 6]  # Happy, Angry, Sad, Neutral
    chosen_indices = [int(np.random.choice(np.where(labels == c)[0])) for c in target_classes]

    dataset_results = []
    ds_correct = 0

    for i, idx in enumerate(chosen_indices, 1):
        img_arr = raw_imgs[idx]
        true_name = CLASS_NAMES[int(labels[idx])].upper()

        save_path = OUTPUT_DIR / f"dataset_random_{i}_{true_name.lower()}.png"
        Image.fromarray(img_arr).resize((112, 112), Image.Resampling.NEAREST).save(save_path)

        pred_name, conf, prob_dict = predict_img_arr(model, device, img_arr)
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
            "Saved Image": str(save_path),
        })

    ds_df = pd.DataFrame(dataset_results)
    print(ds_df[["Sample", "True Emotion", "Predicted Emotion", "Confidence", "Result", "Top Probabilities"]].to_string(index=False))
    print(f"\nDataset 4-Sample Accuracy: {ds_correct}/4 ({ds_correct/4*100:.1f}%)\n")

    print("=" * 70)
    print("2. EVALUATION ON 4 REAL-WORLD INTERNET IMAGES (LIVE DOWNLOAD & FACE CROP)")
    print("=" * 70)

    internet_results = []
    net_correct = 0
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

    for item in INTERNET_IMAGE_URLS:
        print(f"Downloading & detecting face for: {item['name']} ({item['expected_expression']})...")
        try:
            req = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw_data = resp.read()

            pil_orig = Image.open(io.BytesIO(raw_data))
            orig_save_path = OUTPUT_DIR / f"{item['name']}_orig.jpg"
            pil_orig.save(orig_save_path)

            face_48 = crop_portrait_face_pil(pil_orig)
            crop_save_path = OUTPUT_DIR / f"{item['name']}_crop48.png"
            Image.fromarray(face_48).resize((112, 112), Image.Resampling.NEAREST).save(crop_save_path)

            pred_name, conf, prob_dict = predict_img_arr(model, device, face_48)
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
                "Description": item["description"],
                "Saved Face Crop": str(crop_save_path),
            })
        except Exception as e:
            print(f"Error fetching/processing {item['url']}: {e}")
            continue

    net_df = pd.DataFrame(internet_results)
    if not net_df.empty:
        print("\n" + net_df[["Image Name", "Expected Emotion", "Predicted Emotion", "Confidence", "Result", "Top Probabilities"]].to_string(index=False))
        print(f"\nInternet 4-Sample Accuracy: {net_correct}/{len(internet_results)} ({net_correct/max(len(internet_results),1)*100:.1f}%)")

    # Save summary report
    summary = {
        "dataset_samples": dataset_results,
        "dataset_accuracy_pct": ds_correct / 4.0 * 100.0,
        "internet_samples": internet_results,
        "internet_accuracy_pct": net_correct / max(len(internet_results), 1) * 100.0,
    }

    with open(OUTPUT_DIR / "random_and_internet_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_evaluation()
