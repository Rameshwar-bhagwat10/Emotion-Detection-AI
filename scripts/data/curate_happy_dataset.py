"""Script to curate, convert, and quality-filter Happy facial images from AffectNet Short dataset."""

from __future__ import annotations

import io
import os
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

OUTPUT_DIR = ROOT_DIR / "data" / "interim" / "happy_curated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_FILES = [
    ROOT_DIR / "data" / "interim" / "affectnet_short_val.parquet",
    ROOT_DIR / "data" / "interim" / "affectnet_short_train.parquet",
]

AFFECTNET_HAPPY_LABEL = 3


def check_sharpness(img_gray: np.ndarray, threshold: float = 20.0) -> bool:
    """Check image sharpness using Laplacian variance."""
    val = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return bool(val >= threshold)


def curate_happy_from_parquets(target_count: int = 3000) -> list[dict]:
    # Clean output dir
    for f in OUTPUT_DIR.glob("*.png"):
        f.unlink()

    collected = []

    for p_path in PARQUET_FILES:
        if not p_path.exists():
            continue
        print(f"Reading {p_path.name}...")
        try:
            df = pd.read_parquet(p_path)
        except Exception as e:
            print(f"  Skipping {p_path.name}: {e}")
            continue

        print(f"  Rows in {p_path.name}: {len(df)}. Labels:\n{df['label'].value_counts().to_dict()}")

        happy_mask = (df['label'] == AFFECTNET_HAPPY_LABEL)
        happy_rows = df[happy_mask]
        print(f"  Found {len(happy_rows)} Happy rows in {p_path.name}.")

        for _, row in happy_rows.iterrows():
            if len(collected) >= target_count:
                break
            try:
                img_data = row['image']
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    raw_bytes = img_data['bytes']
                elif isinstance(img_data, bytes):
                    raw_bytes = img_data
                else:
                    continue

                pil_img = Image.open(io.BytesIO(raw_bytes)).convert('L')
                np_gray = np.array(pil_img)
                h, w = np_gray.shape[:2]
                if h < 32 or w < 32:
                    continue

                if not check_sharpness(np_gray, threshold=15.0):
                    continue

                # Native FER resolution 48x48
                img_48 = cv2.resize(np_gray, (48, 48), interpolation=cv2.INTER_AREA)

                idx = len(collected)
                filename = f"happy_cand_{idx:05d}.png"
                filepath = OUTPUT_DIR / filename
                Image.fromarray(img_48).save(filepath)

                collected.append({
                    "id": idx,
                    "filename": filename,
                    "source": "affectnet_short",
                    "original_label": "happy",
                    "orig_w": w,
                    "orig_h": h,
                })
                if len(collected) % 500 == 0:
                    print(f"  Curated {len(collected)} clean Happy face crops...")
            except Exception:
                continue

    print(f"Total curated Happy images saved to {OUTPUT_DIR}: {len(collected)}")
    meta_df = pd.DataFrame(collected)
    meta_df.to_csv(OUTPUT_DIR / "candidates_metadata.csv", index=False)
    return collected


def main():
    curate_happy_from_parquets(target_count=3000)


if __name__ == "__main__":
    main()
