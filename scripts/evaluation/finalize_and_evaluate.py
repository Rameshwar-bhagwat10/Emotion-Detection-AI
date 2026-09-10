"""Finalize Model V2 Optimized artifacts, export ONNX, and run comprehensive benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score, confusion_matrix

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

MODELS_DIR = ROOT_DIR / "models" / "v2_optimized"
REPORTS_DIR = ROOT_DIR / "reports" / "tri_emotions"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def finalize_and_evaluate():
    device = torch.device("cpu")
    weights_path = MODELS_DIR / "model.pt"

    # 1. Load trained optimized model
    model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device)
    model.eval()

    # 2. Export clean ONNX
    dummy_input = torch.randn(1, 3, 112, 112, dtype=torch.float32)
    onnx_path = MODELS_DIR / "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=18,
    )
    print(f"Exported clean ONNX model to {onnx_path}")

    # 3. Comprehensive Evaluation on untouched Test Set (3,589 images)
    test_store = TensorDataStore("test")
    num_samples = test_store.num_samples
    all_preds = []
    all_targets = []
    all_probs = []

    mean_dev = IMAGENET_MEAN.to(device)
    std_dev = IMAGENET_STD.to(device)

    with torch.inference_mode():
        for i in range(0, num_samples, 128):
            batch_imgs = test_store.images[i : i + 128].to(device)
            batch_lbls = test_store.labels[i : i + 128].numpy()

            b_112 = F.interpolate(batch_imgs, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            b_norm = (b_112 - mean_dev) / std_dev

            logits = model(b_norm)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_preds.append(preds)
            all_targets.append(batch_lbls)
            all_probs.append(probs)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    acc = float(np.mean(all_preds == all_targets))
    bal_acc = float(balanced_accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))

    rep = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds).tolist()

    # 4. Save metadata
    metadata = {
        "model_name": "emotion-resnet18-cbam-optimized",
        "version": "v2_optimized",
        "description": "Optimized production model with enhanced Fear, Sad, and Angry recognition accuracy via teacher knowledge distillation and curated AffectNet training.",
        "input_size": [112, 112],
        "input_channels": 3,
        "classes": {str(i): c for i, c in enumerate(CLASS_NAMES)},
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "test_metrics": {
            "overall_accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "angry": {
                "precision": round(rep["angry"]["precision"], 4),
                "recall": round(rep["angry"]["recall"], 4),
                "f1_score": round(rep["angry"]["f1-score"], 4),
                "support": rep["angry"]["support"],
            },
            "disgust": {
                "precision": round(rep["disgust"]["precision"], 4),
                "recall": round(rep["disgust"]["recall"], 4),
                "f1_score": round(rep["disgust"]["f1-score"], 4),
                "support": rep["disgust"]["support"],
            },
            "fear": {
                "precision": round(rep["fear"]["precision"], 4),
                "recall": round(rep["fear"]["recall"], 4),
                "f1_score": round(rep["fear"]["f1-score"], 4),
                "support": rep["fear"]["support"],
            },
            "happy": {
                "precision": round(rep["happy"]["precision"], 4),
                "recall": round(rep["happy"]["recall"], 4),
                "f1_score": round(rep["happy"]["f1-score"], 4),
                "support": rep["happy"]["support"],
            },
            "sad": {
                "precision": round(rep["sad"]["precision"], 4),
                "recall": round(rep["sad"]["recall"], 4),
                "f1_score": round(rep["sad"]["f1-score"], 4),
                "support": rep["sad"]["support"],
            },
            "surprise": {
                "precision": round(rep["surprise"]["precision"], 4),
                "recall": round(rep["surprise"]["recall"], 4),
                "f1_score": round(rep["surprise"]["f1-score"], 4),
                "support": rep["surprise"]["support"],
            },
            "neutral": {
                "precision": round(rep["neutral"]["precision"], 4),
                "recall": round(rep["neutral"]["recall"], 4),
                "f1_score": round(rep["neutral"]["f1-score"], 4),
                "support": rep["neutral"]["support"],
            },
        },
        "efficiency": {
            "model_size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 2),
            "total_parameters": sum(p.numel() for p in model.parameters()),
        },
        "status": "production_promoted",
    }

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(REPORTS_DIR / "final_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metadata["test_metrics"], "confusion_matrix": cm}, f, indent=2)

    print("\nBENCHMARK COMPLETE:")
    print(f"Overall Accuracy:  {acc*100:.2f}% (Baseline V2 was 62.66%)")
    print(f"Balanced Accuracy: {bal_acc*100:.2f}% (Baseline V2 was 64.63%)")
    print(f"Macro F1-Score:    {macro_f1*100:.2f}% (Baseline V2 was 58.52%)")
    print(f"Angry F1-Score:    {rep['angry']['f1-score']*100:.2f}% (Baseline was 54.00%)")
    print(f"Fear F1-Score:     {rep['fear']['f1-score']*100:.2f}% (Baseline was 41.50%)")
    print(f"Sad F1-Score:      {rep['sad']['f1-score']*100:.2f}% (Baseline was 51.00%)")
    print(f"Happy F1-Score:    {rep['happy']['f1-score']*100:.2f}% (Baseline was 85.82%)")
    print(f"Surprise F1-Score: {rep['surprise']['f1-score']*100:.2f}% (Baseline was 78.69%)")
    print(f"Neutral F1-Score:  {rep['neutral']['f1-score']*100:.2f}% (Baseline was 57.85%)")
    print(f"Disgust F1-Score:  {rep['disgust']['f1-score']*100:.2f}% (Baseline was 67.80%)")

    # 5. Re-evaluate the 7 Random Person Images with the Optimized Model
    print("\nRe-evaluating 7 random person images with Optimized Model...")
    sample_res = []
    sample_correct = 0
    raw_imgs = (test_store.images.squeeze(1).numpy() * 255.0).astype(np.uint8)

    # Use same seed 42
    np.random.seed(42)
    selected_indices = []
    for class_idx in range(7):
        matching = np.where(all_targets == class_idx)[0]
        selected_indices.append(int(np.random.choice(matching)))
    np.random.shuffle(selected_indices)

    for i, idx in enumerate(selected_indices, 1):
        img_np = raw_imgs[idx]
        true_label = CLASS_NAMES[int(all_targets[idx])]
        pred_label = CLASS_NAMES[int(all_preds[idx])]
        probs = all_probs[idx]
        conf = float(probs[int(all_preds[idx])])

        is_correct = (pred_label == true_label)
        if is_correct:
            sample_correct += 1

        prob_breakdown = {CLASS_NAMES[k]: round(float(probs[k]) * 100, 1) for k in range(7)}
        sample_res.append({
            "Sample": i,
            "True Expression": true_label.upper(),
            "Predicted Expression": pred_label.upper(),
            "Confidence": f"{conf*100:.1f}%",
            "Result": "CORRECT" if is_correct else "INCORRECT",
            "Top Probabilities": ", ".join([f"{k}: {v}%" for k, v in sorted(prob_breakdown.items(), key=lambda x: x[1], reverse=True)[:3]]),
        })

    sample_df = pd.DataFrame(sample_res)
    print("\n" + sample_df.to_string(index=False))
    print(f"\nRandom Sample Accuracy: {sample_correct}/7 ({sample_correct/7*100:.2f}%)")

    sample_df.to_csv(REPORTS_DIR / "random_7_sample_retest.csv", index=False)


if __name__ == "__main__":
    finalize_and_evaluate()
