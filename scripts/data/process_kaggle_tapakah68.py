"""Ingest and preprocess Kaggle tapakah68/facial-emotion-recognition dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile
import cv2
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

DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    "anger": 0, "angry": 0, "ang": 0,
    "disgust": 1, "disgusted": 1, "dis": 1,
    "fear": 2, "fearful": 2, "fea": 2,
    "happy": 3, "happiness": 3, "joy": 3, "hap": 3,
    "sad": 4, "sadness": 4,
    "surprise": 5, "surprised": 5, "sur": 5,
    "neutral": 6, "neutrality": 6, "contempt": 6, "neu": 6,
}

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def find_zip_file() -> Path | None:
    candidates = [
        DATA_RAW / "facial-emotion-recognition.zip",
        DATA_RAW / "archive.zip",
        Path("C:/Users/RAMESHWAR/Downloads/facial-emotion-recognition.zip"),
        Path("C:/Users/RAMESHWAR/Downloads/archive.zip"),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 10 * 1024 * 1024:  # At least 10MB
            return p
    # Search downloads directory for any zip containing emotion
    dl_dir = Path("C:/Users/RAMESHWAR/Downloads")
    if dl_dir.exists():
        for z in dl_dir.glob("*.zip"):
            if "emotion" in z.name.lower() and z.stat().st_size > 10 * 1024 * 1024:
                return z
    return None


def extract_and_process():
    zip_path = find_zip_file()
    if not zip_path:
        print("ERROR: facial-emotion-recognition.zip not found in data/raw/ or Downloads/.")
        print("Please download it from https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition")
        return False

    dest_dir = DATA_RAW / "kaggle_tapakah68"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path} to {dest_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    print("Extraction complete.")

    # Find all images
    image_files = list(dest_dir.rglob("*.jpg")) + list(dest_dir.rglob("*.png")) + list(dest_dir.rglob("*.jpeg"))
    print(f"Found {len(image_files)} image files.")

    if len(image_files) == 0:
        print("No images found in extracted archive.")
        return False

    processed_images = []
    processed_labels = []
    class_counts = {c: 0 for c in CLASS_NAMES}

    for img_path in image_files:
        # Determine emotion from file name or parent folder name
        label_idx = None
        lower_path = str(img_path).lower()

        for key, idx in CLASS_MAP.items():
            if key in img_path.stem.lower() or key in img_path.parent.name.lower():
                label_idx = idx
                break

        if label_idx is None:
            continue

        try:
            pil_img = Image.open(img_path).convert("L")
            # Face crop (center square crop if full resolution portrait)
            w, h = pil_img.size
            min_dim = min(w, h)
            cy, cx = int(h * 0.45), int(w * 0.50)
            half = int(min_dim * 0.40)
            x0, y0 = max(0, cx - half), max(0, cy - half)
            x1, y1 = min(w, cx + half), min(h, cy + half)
            face = pil_img.crop((x0, y0, x1, y1)).resize((48, 48), Image.Resampling.BILINEAR)

            arr = np.array(face, dtype=np.uint8)
            processed_images.append(arr)
            processed_labels.append(label_idx)
            class_counts[CLASS_NAMES[label_idx]] += 1
        except Exception as e:
            continue

    if len(processed_images) == 0:
        print("Could not process any images.")
        return False

    imgs_np = np.stack(processed_images)  # [N, 48, 48]
    lbls_np = np.array(processed_labels, dtype=np.int64)  # [N]

    output_path = DATA_PROCESSED / "kaggle_tapakah68_curated.npz"
    np.savez_compressed(output_path, images=imgs_np, labels=lbls_np)
    print(f"Successfully processed {len(imgs_np)} images into {output_path}!")
    print("Class Distribution:")
    for k, v in class_counts.items():
        print(f"  {k.upper():<9}: {v}")

    return True


if __name__ == "__main__":
    extract_and_process()
