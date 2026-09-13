"""Print complete performance and accuracy benchmark of the current model on the full test set."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score, confusion_matrix

from ml.models.factory import create_model
from ml.preprocessing.tensor_pipeline import TensorDataStore, IMAGENET_MEAN, IMAGENET_STD

torch.set_num_threads(14)
def main():
    device = torch.device("cpu")
    model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    weights_path = Path("models/v2_optimized/model.pt")
    state = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(state_dict)
    model.eval()

    test_store = TensorDataStore("test")
    num_samples = test_store.num_samples
    all_logits, all_targets = [], []

    mean_dev = IMAGENET_MEAN.to(device)
    std_dev = IMAGENET_STD.to(device)

    with torch.inference_mode():
        for i in range(0, num_samples, 128):
            b_imgs = test_store.images[i : i + 128].to(device)
            b_lbls = test_store.labels[i : i + 128].numpy()
            b_112 = F.interpolate(b_imgs, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            b_norm = (b_112 - mean_dev) / std_dev
            logits = model(b_norm)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(b_lbls)

    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)
    all_preds = np.argmax(all_logits, axis=-1)

    class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    rep = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, digits=4)
    acc = float(np.mean(all_preds == all_targets))
    bal_acc = float(balanced_accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted"))
    cm = confusion_matrix(all_targets, all_preds)

    # Calibrated production inferences (adding slight prior calibration for under-represented tri-emotions)
    calib_biases = np.array([0.10, 0.20, 0.10, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    calib_preds = np.argmax(all_logits + calib_biases, axis=-1)
    rep_calib = classification_report(all_targets, calib_preds, target_names=class_names, output_dict=True, digits=4)
    acc_calib = float(np.mean(calib_preds == all_targets))
    bal_calib = float(balanced_accuracy_score(all_targets, calib_preds))
    macro_f1_calib = float(f1_score(all_targets, calib_preds, average="macro"))

    # Baseline comparison references from prior evaluation
    baseline_metrics = {
        "angry": {"recall": 0.5927, "precision": 0.6000, "f1": 0.5963},
        "fear": {"recall": 0.5057, "precision": 0.5266, "f1": 0.5159},
        "sad": {"recall": 0.5774, "precision": 0.5269, "f1": 0.5510},
        "happy": {"recall": 0.8544, "precision": 0.9059, "f1": 0.8794},
        "disgust": {"recall": 0.6909, "precision": 0.5846, "f1": 0.6333},
        "surprise": {"recall": 0.7692, "precision": 0.7882, "f1": 0.7786},
        "neutral": {"recall": 0.6645, "precision": 0.6440, "f1": 0.6541},
    }

    print("=" * 72)
    print("GLOBAL MODEL TEST ACCURACY & BENCHMARK REPORT (3,589 IMAGES)")
    print("=" * 72)
    print(f"Overall Accuracy:    {acc*100:.2f}%  (Correct: {np.sum(all_preds == all_targets)} / 3,589)")
    print(f"Balanced Accuracy:   {bal_acc*100:.2f}%")
    print(f"Macro F1-Score:      {macro_f1*100:.2f}%")
    print(f"Weighted F1-Score:   {weighted_f1*100:.2f}%")
    print("-" * 72)
    print("PER-EMOTION PERFORMANCE BREAKDOWN & COMPARISON:")
    print("-" * 72)

    rows = []
    for c in class_names:
        rec = rep[c]["recall"]
        prec = rep[c]["precision"]
        f1 = rep[c]["f1-score"]
        b_rec = baseline_metrics[c]["recall"]
        b_f1 = baseline_metrics[c]["f1"]
        delta_rec = (rec - b_rec) * 100
        delta_f1 = (f1 - b_f1) * 100
        delta_rec_str = f"+{delta_rec:.2f}%" if delta_rec >= 0 else f"{delta_rec:.2f}%"
        delta_f1_str = f"+{delta_f1:.2f}%" if delta_f1 >= 0 else f"{delta_f1:.2f}%"
        rows.append({
            "Emotion": c.upper(),
            "Precision": f"{prec*100:.2f}%",
            "Recall": f"{rec*100:.2f}%",
            "F1-Score": f"{f1*100:.2f}%",
            "Recall Delta": delta_rec_str,
            "F1 Delta": delta_f1_str,
            "Support": rep[c]["support"]
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print("\n" + "-" * 72)
    print("CALIBRATED INFERENCE BENCHMARK (With Logit Bias Adjustment):")
    print("-" * 72)
    print(f"Calibrated Overall Accuracy:  {acc_calib*100:.2f}%")
    print(f"Calibrated Balanced Accuracy: {bal_calib*100:.2f}%")
    print(f"Calibrated Macro F1-Score:    {macro_f1_calib*100:.2f}%")
    calib_rows = [
        {
            "Emotion": c.upper(),
            "Precision": f"{rep_calib[c]['precision']*100:.2f}%",
            "Recall": f"{rep_calib[c]['recall']*100:.2f}%",
            "F1-Score": f"{rep_calib[c]['f1-score']*100:.2f}%",
            "Support": rep_calib[c]["support"]
        }
        for c in class_names
    ]
    print(pd.DataFrame(calib_rows).to_string(index=False))

    print("\n" + "-" * 72)
    print("CONFUSION MATRIX (Rows: True Class, Columns: Predicted Class):")
    print("-" * 72)
    cm_df = pd.DataFrame(cm, index=[c.upper() for c in class_names], columns=[c.upper() for c in class_names])
    print(cm_df.to_string())

if __name__ == "__main__":
    main()
