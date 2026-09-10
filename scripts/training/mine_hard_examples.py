"""Hard Example Mining for Model V3: Positive Hard Examples (Happy FN) and Negative Hard Examples (Happy FP)."""

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
from ml.preprocessing.tensor_pipeline import TensorDataStore, FastTensorDataLoader

REPORTS_DIR = ROOT_DIR / "reports" / "v3"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
HAPPY_IDX = 3


def mine_hard_examples() -> dict:
    print("=" * 60)
    print("MINING POSITIVE & NEGATIVE HARD EXAMPLES USING MODEL V2 ON TRAINING DATA")
    print("=" * 60)

    device = torch.device("cpu")
    model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    v2_weights = ROOT_DIR / "models" / "v2" / "model.pt"
    state_dict = torch.load(v2_weights, map_location=device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    train_store = TensorDataStore("train")
    y_true = train_store.labels.numpy()
    loader = FastTensorDataLoader(
        train_store,
        batch_size=256,
        shuffle=False,
        input_size=(112, 112),
        channels=3,
        augment=False,
    )

    all_probs = []
    print(f"Running inference on {train_store.num_samples} training images...")
    with torch.inference_mode():
        for imgs, _ in loader:
            logits = model(imgs.to(device))
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    preds = np.argmax(all_probs, axis=-1)
    confs = np.max(all_probs, axis=-1)
    happy_probs = all_probs[:, HAPPY_IDX]

    # 1. Positive Hard Examples (True Happy, but misclassified or low confidence)
    happy_mask = (y_true == HAPPY_IDX)
    pos_hard_mask = happy_mask & ((preds != HAPPY_IDX) | (happy_probs < 0.50))
    pos_hard_indices = np.where(pos_hard_mask)[0]

    # 2. Negative Hard Examples (True NOT Happy, but predicted Happy or high Happy probability)
    not_happy_mask = (y_true != HAPPY_IDX)
    neg_hard_mask = not_happy_mask & ((preds == HAPPY_IDX) | (happy_probs >= 0.25))
    neg_hard_indices = np.where(neg_hard_mask)[0]

    print(f"Total Training Samples: {len(y_true)}")
    print(f"Total True Happy Samples: {np.sum(happy_mask)}")
    print(f"Positive Hard Examples (Happy FN / Ambiguous): {len(pos_hard_indices)}")
    print(f"Negative Hard Examples (Non-Happy FP / High Risk): {len(neg_hard_indices)}")

    # Confusion breakdown of Positive Hard Examples
    pos_hard_confusions = {}
    for idx in pos_hard_indices:
        pred_label = CLASS_NAMES[preds[idx]]
        pos_hard_confusions[pred_label] = pos_hard_confusions.get(pred_label, 0) + 1

    # Confusion breakdown of Negative Hard Examples
    neg_hard_confusions = {}
    for idx in neg_hard_indices:
        true_label = CLASS_NAMES[y_true[idx]]
        neg_hard_confusions[true_label] = neg_hard_confusions.get(true_label, 0) + 1

    # Create sample weight tensor for training
    # Standard sample weight = 1.0
    # Positive hard happy sample = 2.0 (boosted exposure)
    # Negative hard sample = 1.5 (boosted focus to avoid false alarms)
    sample_weights = np.ones(len(y_true), dtype=np.float32)
    sample_weights[pos_hard_indices] = 2.0
    sample_weights[neg_hard_indices] = 1.5

    # Normalize weights so mean is 1.0
    sample_weights = sample_weights / np.mean(sample_weights)

    # Save to NPZ
    np.savez_compressed(
        PROCESSED_DIR / "fer2013_hard_weights.npz",
        sample_weights=sample_weights,
        pos_hard_indices=pos_hard_indices,
        neg_hard_indices=neg_hard_indices,
    )

    mining_results = {
        "total_train_samples": int(len(y_true)),
        "total_happy_samples": int(np.sum(happy_mask)),
        "positive_hard_happy_count": int(len(pos_hard_indices)),
        "positive_hard_confusions": pos_hard_confusions,
        "negative_hard_count": int(len(neg_hard_indices)),
        "negative_hard_confusions": neg_hard_confusions,
        "hard_weights_file": str(PROCESSED_DIR / "fer2013_hard_weights.npz"),
    }

    with open(REPORTS_DIR / "hard_examples_analysis.json", "w", encoding="utf-8") as f:
        json.dump(mining_results, f, indent=2)

    print(f"Saved hard examples analysis to {REPORTS_DIR / 'hard_examples_analysis.json'}")
    return mining_results


def main():
    mine_hard_examples()


if __name__ == "__main__":
    main()
