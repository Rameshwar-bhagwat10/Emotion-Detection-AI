"""Script to pick 7 random facial images with random expressions, test Model V2, and report accuracy."""

from __future__ import annotations

import json
from pathlib import Path
import sys
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

SAMPLE_EXPORT_DIR = ROOT_DIR / "reports" / "random_eval_samples"
SAMPLE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def run_random_7_sample_eval(seed: int = 42) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    test_store = TensorDataStore("test")
    raw_imgs = (test_store.images.squeeze(1).numpy() * 255.0).astype(np.uint8)
    labels = test_store.labels.numpy()

    # Load Model V2
    model_v2 = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    v2_weights = ROOT_DIR / "models" / "v2" / "model.pt"
    s_v2 = torch.load(v2_weights, map_location=device)
    model_v2.load_state_dict(s_v2["model_state_dict"] if "model_state_dict" in s_v2 else s_v2)
    model_v2.to(device)
    model_v2.eval()

    # Pick 1 random person image for each of the 7 distinct emotion classes
    selected_indices = []
    for class_idx in range(7):
        matching_indices = np.where(labels == class_idx)[0]
        chosen = int(np.random.choice(matching_indices))
        selected_indices.append(chosen)

    # Shuffle the 7 samples for random ordering
    np.random.shuffle(selected_indices)

    results = []
    correct_count = 0

    print("=" * 70)
    print("TESTING MODEL V2 ON 7 RANDOM PERSON FACIAL IMAGES ACROSS EXPRESSIONS")
    print("=" * 70)

    for i, idx in enumerate(selected_indices, 1):
        img_np = raw_imgs[idx]
        true_label_idx = int(labels[idx])
        true_label_name = CLASS_NAMES[true_label_idx]

        # Save image for visual verification
        img_filename = f"sample_{i}_{true_label_name}.png"
        img_path = SAMPLE_EXPORT_DIR / img_filename
        Image.fromarray(img_np).resize((112, 112), Image.Resampling.NEAREST).save(img_path)

        # Prepare tensor for Model V2
        t_img = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float() / 255.0  # [1, 1, 48, 48]
        t_112 = F.interpolate(t_img, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        t_norm = (t_112 - IMAGENET_MEAN) / IMAGENET_STD

        with torch.inference_mode():
            logits = model_v2(t_norm)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
            pred_idx = int(np.argmax(probs))
            pred_name = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])

        is_correct = (pred_idx == true_label_idx)
        if is_correct:
            correct_count += 1

        prob_dict = {CLASS_NAMES[k]: round(float(probs[k]) * 100, 2) for k in range(7)}

        results.append({
            "sample_num": i,
            "test_dataset_index": idx,
            "saved_image": str(img_path),
            "true_expression": true_label_name.upper(),
            "predicted_expression": pred_name.upper(),
            "confidence_pct": round(confidence * 100, 2),
            "is_correct": is_correct,
            "status": "PASS" if is_correct else "FAIL",
            "probabilities_pct": prob_dict,
        })

    sample_accuracy = (correct_count / len(results)) * 100.0

    print(f"\nCompleted evaluation of {len(results)} random person images.")
    print(f"Sample Accuracy: {correct_count}/{len(results)} ({sample_accuracy:.2f}%)\n")

    summary = {
        "model_evaluated": "Model V2 (ResNet-18 + CBAM Production)",
        "num_samples_evaluated": len(results),
        "correct_predictions": correct_count,
        "sample_accuracy_pct": round(sample_accuracy, 2),
        "production_test_benchmark": {
            "overall_accuracy_pct": 62.66,
            "balanced_accuracy_pct": 64.63,
            "macro_f1_pct": 58.52,
            "weighted_f1_pct": 61.59,
            "total_test_images": 3589,
        },
        "samples": results,
    }

    # Save to JSON
    json_path = SAMPLE_EXPORT_DIR / "random_7_eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    res = run_random_7_sample_eval(seed=42)
    df = pd.DataFrame([
        {
            "Sample": r["sample_num"],
            "True Expression": r["true_expression"],
            "Predicted Expression": r["predicted_expression"],
            "Confidence": f"{r['confidence_pct']:.1f}%",
            "Result": "CORRECT" if r["is_correct"] else "INCORRECT",
            "Top Probabilities": ", ".join([f"{k}: {v:.1f}%" for k, v in sorted(r["probabilities_pct"].items(), key=lambda x: x[1], reverse=True)[:3]]),
        }
        for r in res["samples"]
    ])
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
