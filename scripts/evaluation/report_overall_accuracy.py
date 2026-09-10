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
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    test_store = TensorDataStore("test")
    num_samples = test_store.num_samples
    all_preds, all_targets = [], []

    mean_dev = IMAGENET_MEAN.to(device)
    std_dev = IMAGENET_STD.to(device)

    with torch.inference_mode():
        for i in range(0, num_samples, 128):
            b_imgs = test_store.images[i : i + 128].to(device)
            b_lbls = test_store.labels[i : i + 128].numpy()
            b_112 = F.interpolate(b_imgs, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            b_norm = (b_112 - mean_dev) / std_dev
            logits = model(b_norm)
            all_preds.append(logits.argmax(dim=-1).numpy())
            all_targets.append(b_lbls)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    rep = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, digits=4)
    acc = float(np.mean(all_preds == all_targets))
    bal_acc = float(balanced_accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted"))
    cm = confusion_matrix(all_targets, all_preds)

    print("=" * 65)
    print("GLOBAL MODEL TEST ACCURACY & BENCHMARK REPORT (3,589 IMAGES)")
    print("=" * 65)
    print(f"Overall Accuracy:    {acc*100:.2f}%  (Correct: {np.sum(all_preds == all_targets)} / 3,589)")
    print(f"Balanced Accuracy:   {bal_acc*100:.2f}%")
    print(f"Macro F1-Score:      {macro_f1*100:.2f}%")
    print(f"Weighted F1-Score:   {weighted_f1*100:.2f}%")
    print("-" * 65)
    print("PER-EMOTION BREAKDOWN:")
    print("-" * 65)

    df = pd.DataFrame([
        {
            "Emotion": c.upper(),
            "Precision": f"{rep[c]['precision']*100:.2f}%",
            "Recall": f"{rep[c]['recall']*100:.2f}%",
            "F1-Score": f"{rep[c]['f1-score']*100:.2f}%",
            "Support": rep[c]["support"]
        }
        for c in class_names
    ])
    print(df.to_string(index=False))

    print("\n" + "-" * 65)
    print("CONFUSION MATRIX (Rows: True Class, Columns: Predicted Class):")
    print("-" * 65)
    cm_df = pd.DataFrame(cm, index=[c.upper() for c in class_names], columns=[c.upper() for c in class_names])
    print(cm_df.to_string())

if __name__ == "__main__":
    main()
