"""Filter curated Angry, Fear, and Sad samples using Teacher Model V2 confidence gating."""

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
from ml.preprocessing.tensor_pipeline import IMAGENET_MEAN, IMAGENET_STD

CURATED_DIR = ROOT_DIR / "data" / "interim" / "tri_emotions_curated"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "tri_emotions_curated.npz"
REPORTS_DIR = ROOT_DIR / "reports" / "tri_emotions"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
TARGET_EMOTION_MAP = {
    "angry": 0,
    "fear": 2,
    "sad": 4,
}


def filter_with_teacher(confidence_thresh: float = 0.30) -> dict:
    device = torch.device("cpu")
    model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    v2_weights = ROOT_DIR / "models" / "v2" / "model.pt"
    s = torch.load(v2_weights, map_location=device)
    model.load_state_dict(s["model_state_dict"] if "model_state_dict" in s else s)
    model.to(device)
    model.eval()

    cand_files = sorted(list(CURATED_DIR.glob("*.png")))
    print(f"Filtering {len(cand_files)} leak-free candidates with Teacher Model V2...")

    accepted_imgs = []
    accepted_lbls = []
    accepted_weights = []
    log_rows = []

    for fpath in cand_files:
        prefix = fpath.stem.split("_")[0]
        if prefix not in TARGET_EMOTION_MAP:
            continue
        target_idx = TARGET_EMOTION_MAP[prefix]

        img = Image.open(fpath).convert("L")
        arr_48 = np.array(img, dtype=np.uint8)

        # Tensor preprocessing
        t_img = torch.from_numpy(arr_48).unsqueeze(0).unsqueeze(0).float() / 255.0  # [1, 1, 48, 48]
        t_112 = F.interpolate(t_img, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        t_norm = (t_112 - IMAGENET_MEAN) / IMAGENET_STD

        with torch.inference_mode():
            logits = model(t_norm)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

        target_prob = float(probs[target_idx])
        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx]
        top2_indices = np.argsort(probs)[-2:]

        # Agreement or Top-2 with reasonable confidence
        passed = (pred_idx == target_idx) or ((target_idx in top2_indices) and (target_prob >= confidence_thresh))

        if passed:
            accepted_imgs.append(arr_48)
            accepted_lbls.append(target_idx)
            # Higher weight for confident samples
            sample_w = 1.0 + target_prob
            accepted_weights.append(sample_w)

        log_rows.append({
            "filename": fpath.name,
            "target_emotion": prefix,
            "target_class": target_idx,
            "pred_emotion": pred_name,
            "pred_class": pred_idx,
            "target_prob": round(target_prob, 4),
            "passed_gate": passed,
        })

    accepted_imgs_np = np.stack(accepted_imgs, axis=0)  # [N, 48, 48] uint8
    accepted_lbls_np = np.array(accepted_lbls, dtype=np.int64)  # [N]
    accepted_weights_np = np.array(accepted_weights, dtype=np.float32)  # [N]

    np.savez_compressed(
        OUTPUT_FILE,
        images=accepted_imgs_np,
        labels=accepted_lbls_np,
        weights=accepted_weights_np,
    )

    counts = {CLASS_NAMES[k]: int(np.sum(accepted_lbls_np == k)) for k in [0, 2, 4]}

    summary = {
        "total_evaluated": len(cand_files),
        "total_accepted": len(accepted_imgs),
        "acceptance_rate_pct": round(len(accepted_imgs) / max(len(cand_files), 1) * 100, 2),
        "class_counts": counts,
        "saved_path": str(OUTPUT_FILE),
    }

    with open(REPORTS_DIR / "tri_emotion_teacher_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(log_rows).to_csv(REPORTS_DIR / "tri_emotion_teacher_log.csv", index=False)

    print("\nTEACHER FILTERING COMPLETE:")
    print(f"Accepted {len(accepted_imgs)} / {len(cand_files)} ({summary['acceptance_rate_pct']}%)")
    print(f"Counts -> Angry: {counts['angry']}, Fear: {counts['fear']}, Sad: {counts['sad']}")
    print(f"Saved to {OUTPUT_FILE}")
    return summary


def main():
    filter_with_teacher(confidence_thresh=0.25)


if __name__ == "__main__":
    main()
