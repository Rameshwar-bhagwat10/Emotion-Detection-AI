"""Teacher Model V2 Filtering and Confidence Agreement Gating for Model V3 Happy Dataset."""

from __future__ import annotations

import io
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

from ml.models.factory import create_model
from ml.preprocessing.tensor_pipeline import IMAGENET_MEAN, IMAGENET_STD

REPORTS_DIR = ROOT_DIR / "reports" / "v3"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CANDIDATES_DIR = ROOT_DIR / "data" / "interim" / "happy_curated"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
HAPPY_IDX = 3


def filter_happy_candidates_with_teacher(
    conf_threshold: float = 0.70,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    print("=" * 60)
    print(f"TEACHER MODEL V2 INFERENCE & CONFIDENCE AGREEMENT FILTERING (Threshold >= {conf_threshold:.2f})")
    print("=" * 60)

    # 1. Load Model V2
    device = torch.device("cpu")
    model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    v2_checkpoint_path = ROOT_DIR / "models" / "v2" / "model.pt"
    if not v2_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Model V2 weights at {v2_checkpoint_path}")

    state_dict = torch.load(v2_checkpoint_path, map_location=device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Load clean candidates from leakage audit
    leak_audit_path = REPORTS_DIR / "happy_leakage_audit.json"
    if not leak_audit_path.exists():
        raise FileNotFoundError(f"Must run leakage audit first! Missing {leak_audit_path}")

    with open(leak_audit_path, "r", encoding="utf-8") as f:
        leak_audit = json.load(f)

    # Load rejection files to skip
    rejections_csv = REPORTS_DIR / "happy_rejections_audit.csv"
    rejected_files = set()
    if rejections_csv.exists():
        rdf = pd.read_csv(rejections_csv)
        rejected_files = set(rdf["file"].tolist())

    cand_files = sorted([f for f in CANDIDATES_DIR.glob("happy_cand_*.png") if f.name not in rejected_files])
    print(f"Evaluating {len(cand_files)} leak-free candidates with Teacher Model V2...")

    filtering_log = []
    retained_images_48 = []
    retained_labels = []

    mean_tensor = IMAGENET_MEAN.to(device)
    std_tensor = IMAGENET_STD.to(device)

    with torch.inference_mode():
        for f_idx, f_path in enumerate(cand_files):
            img_48 = np.array(Image.open(f_path).convert("L"), dtype=np.uint8)
            # Transform to [1, 3, 112, 112]
            t_img = torch.from_numpy(img_48).unsqueeze(0).unsqueeze(0).float() / 255.0  # [1, 1, 48, 48]
            t_img_112 = F.interpolate(t_img, size=(112, 112), mode="bilinear", align_corners=False)
            t_img_rgb = t_img_112.repeat(1, 3, 1, 1)
            t_img_norm = (t_img_rgb - mean_tensor) / std_tensor

            logits = model(t_img_norm)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            pred_class_idx = int(np.argmax(probs))
            pred_class_name = CLASS_NAMES[pred_class_idx]
            happy_prob = float(probs[HAPPY_IDX])
            confidence = float(np.max(probs))

            # Sorted top 2
            top2_indices = np.argsort(probs)[::-1][:2]
            top2_str = f"{CLASS_NAMES[top2_indices[0]]}({probs[top2_indices[0]]:.2f}), {CLASS_NAMES[top2_indices[1]]}({probs[top2_indices[1]]:.2f})"

            # Agreement rule: Original label Happy AND Teacher Pred Happy AND P(Happy) >= conf_threshold
            is_retained = bool(pred_class_idx == HAPPY_IDX and happy_prob >= conf_threshold)

            reason = "Accepted" if is_retained else (
                f"Low Happy probability ({happy_prob:.3f} < {conf_threshold})" if pred_class_idx == HAPPY_IDX else
                f"Teacher predicted '{pred_class_name}' ({confidence:.3f})"
            )

            filtering_log.append({
                "file": f_path.name,
                "original_label": "happy",
                "teacher_pred_class": pred_class_name,
                "teacher_happy_prob": round(happy_prob, 4),
                "teacher_confidence": round(confidence, 4),
                "top2_predictions": top2_str,
                "retained": is_retained,
                "filter_reason": reason,
            })

            if is_retained:
                retained_images_48.append(img_48)
                retained_labels.append(HAPPY_IDX)

    log_df = pd.DataFrame(filtering_log)
    log_df.to_csv(REPORTS_DIR / "happy_teacher_filtering_log.csv", index=False)

    retained_arr = np.array(retained_images_48, dtype=np.uint8)
    retained_lbl_arr = np.array(retained_labels, dtype=np.int64)

    # Save to processed NPZ
    np.savez_compressed(
        PROCESSED_DIR / "happy_v3_curated.npz",
        images=retained_arr,
        labels=retained_lbl_arr,
    )

    teacher_summary = {
        "evaluated_candidates": len(cand_files),
        "retained_high_confidence_happy": len(retained_images_48),
        "acceptance_rate_pct": round(len(retained_images_48) / max(len(cand_files), 1) * 100, 2),
        "min_happy_prob_threshold": conf_threshold,
        "mean_retained_confidence": float(np.mean(log_df[log_df["retained"]]["teacher_confidence"])) if len(retained_images_48) > 0 else 0.0,
        "output_npz_path": str(PROCESSED_DIR / "happy_v3_curated.npz"),
    }

    print("\nTEACHER FILTERING SUMMARY:")
    for k, v in teacher_summary.items():
        print(f"  {k}: {v}")

    with open(REPORTS_DIR / "happy_teacher_summary.json", "w", encoding="utf-8") as f:
        json.dump(teacher_summary, f, indent=2)

    return retained_arr, retained_lbl_arr, log_df


def main():
    filter_happy_candidates_with_teacher(conf_threshold=0.70)


if __name__ == "__main__":
    main()
