"""Curate clean facial crops for Angry, Fear, and Sad emotions from AffectNet dataset."""

from __future__ import annotations

import io
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

OUTPUT_DIR = ROOT_DIR / "data" / "interim" / "tri_emotions_curated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_FILES = [
    ROOT_DIR / "data" / "interim" / "affectnet_short_val.parquet",
    ROOT_DIR / "data" / "interim" / "affectnet_short_train.parquet",
]

# Mapping from AffectNet labels to FER2013 target classes
# AffectNet: 0: anger (FER 0), 5: fear (FER 2), 6: sad (FER 4)
AFFECTNET_MAP = {
    0: ("angry", 0),
    5: ("fear", 2),
    6: ("sad", 4),
}


def check_sharpness(img_gray: np.ndarray, threshold: float = 15.0) -> bool:
    val = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return bool(val >= threshold)


def curate_tri_emotions(max_per_class: int = 1500) -> list[dict]:
    # Clean previous output
    for f in OUTPUT_DIR.glob("*.png"):
        f.unlink()

    counts = {0: 0, 2: 0, 4: 0}  # FER target classes: 0 (angry), 2 (fear), 4 (sad)
    collected = []

    for p_path in PARQUET_FILES:
        if not p_path.exists():
            continue
        print(f"Reading {p_path.name}...")
        df = pd.read_parquet(p_path)

        for _, row in df.iterrows():
            aff_lbl = int(row["label"])
            if aff_lbl not in AFFECTNET_MAP:
                continue

            target_name, target_fer_idx = AFFECTNET_MAP[aff_lbl]
            if counts[target_fer_idx] >= max_per_class:
                continue

            try:
                img_data = row["image"]
                if isinstance(img_data, dict) and "bytes" in img_data:
                    raw_bytes = img_data["bytes"]
                elif isinstance(img_data, bytes):
                    raw_bytes = img_data
                else:
                    continue

                pil_img = Image.open(io.BytesIO(raw_bytes)).convert("L")
                np_gray = np.array(pil_img)
                h, w = np_gray.shape[:2]
                if h < 32 or w < 32:
                    continue

                if not check_sharpness(np_gray, threshold=15.0):
                    continue

                img_48 = cv2.resize(np_gray, (48, 48), interpolation=cv2.INTER_AREA)

                idx = len(collected)
                filename = f"{target_name}_{counts[target_fer_idx]:05d}.png"
                filepath = OUTPUT_DIR / filename
                Image.fromarray(img_48).save(filepath)

                counts[target_fer_idx] += 1
                collected.append({
                    "id": idx,
                    "filename": filename,
                    "target_emotion": target_name,
                    "target_fer_class": target_fer_idx,
                    "affectnet_label": aff_lbl,
                    "orig_w": w,
                    "orig_h": h,
                })

                if len(collected) % 500 == 0:
                    print(f"  Curated {len(collected)} samples: Angry={counts[0]}, Fear={counts[2]}, Sad={counts[4]}")
            except Exception:
                continue

    print(f"\nTotal curated samples saved to {OUTPUT_DIR}: {len(collected)}")
    print(f"Counts per class -> Angry: {counts[0]}, Fear: {counts[2]}, Sad: {counts[4]}")
    meta_df = pd.DataFrame(collected)
    meta_df.to_csv(OUTPUT_DIR / "candidates_metadata.csv", index=False)
    return collected


def main():
    curate_tri_emotions(max_per_class=1500)


if __name__ == "__main__":
    main()
