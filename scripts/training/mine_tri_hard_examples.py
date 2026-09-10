"""Hard Example Miner targeting Fear, Sad, and Angry confusion pairs across FER2013."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
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

OUTPUT_WEIGHTS_FILE = ROOT_DIR / "data" / "processed" / "fer2013_tri_hard_weights.npz"
REPORTS_DIR = ROOT_DIR / "reports" / "tri_emotions"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
TARGET_CLASSES = {0: "angry", 2: "fear", 4: "sad"}


def mine_tri_hard_examples(batch_size: int = 128) -> dict:
    device = torch.device("cpu")
    train_store = TensorDataStore("train")
    n_samples = train_store.num_samples
    labels = train_store.labels.numpy()

    model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    v2_weights = ROOT_DIR / "models" / "v2" / "model.pt"
    s = torch.load(v2_weights, map_location=device)
    model.load_state_dict(s["model_state_dict"] if "model_state_dict" in s else s)
    model.to(device)
    model.eval()

    print(f"Mining hard samples on {n_samples} training images...")
    sample_weights = np.ones(n_samples, dtype=np.float32)
    stats = {
        "hard_angry_fn": 0,
        "hard_fear_fn": 0,
        "hard_sad_fn": 0,
        "hard_fp_on_neutral": 0,
    }

    mean_dev = IMAGENET_MEAN.to(device)
    std_dev = IMAGENET_STD.to(device)

    with torch.inference_mode():
        for i in range(0, n_samples, batch_size):
            b_imgs = train_store.images[i : i + batch_size].to(device)
            b_lbls = labels[i : i + batch_size]

            b_112 = F.interpolate(b_imgs, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            b_norm = (b_112 - mean_dev) / std_dev

            logits = model(b_norm)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            for b_idx in range(len(b_lbls)):
                idx = i + b_idx
                true_lbl = b_lbls[b_idx]
                pred_lbl = preds[b_idx]
                p_true = probs[b_idx, true_lbl]

                # If False Negative on Angry (0), Fear (2), or Sad (4)
                if true_lbl in TARGET_CLASSES and pred_lbl != true_lbl:
                    # Weight proportional to difficulty (1.5 to 2.2)
                    w = 1.5 + (1.0 - p_true) * 0.7
                    sample_weights[idx] = w
                    if true_lbl == 0:
                        stats["hard_angry_fn"] += 1
                    elif true_lbl == 2:
                        stats["hard_fear_fn"] += 1
                    elif true_lbl == 4:
                        stats["hard_sad_fn"] += 1

                # If Neutral (6) was misclassified as Fear/Sad/Angry
                elif true_lbl == 6 and pred_lbl in TARGET_CLASSES:
                    sample_weights[idx] = 1.3
                    stats["hard_fp_on_neutral"] += 1

    # Normalize weights so mean is 1.0
    sample_weights = sample_weights / sample_weights.mean()

    np.savez_compressed(
        OUTPUT_WEIGHTS_FILE,
        sample_weights=sample_weights,
    )

    summary = {
        "total_train_samples": n_samples,
        "hard_samples_stats": stats,
        "saved_path": str(OUTPUT_WEIGHTS_FILE),
    }

    with open(REPORTS_DIR / "tri_hard_examples_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nHARD SAMPLE MINING COMPLETE:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Saved sample weights to {OUTPUT_WEIGHTS_FILE}")
    return summary


def main():
    mine_tri_hard_examples(batch_size=128)


if __name__ == "__main__":
    main()
