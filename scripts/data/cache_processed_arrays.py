"""Cache raw FER2013 CSV data into compact, high-speed NumPy archive files."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_CSV = ROOT_DIR / "data" / "raw" / "fer2013" / "fer2013.csv"
OUT_DIR = ROOT_DIR / "data" / "processed"

EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

SPLIT_MAP = {
    "training": "train",
    "publictest": "val",
    "privatetest": "test",
    "train": "train",
    "val": "val",
    "test": "test",
}


def cache_fer2013() -> None:
    """Parse CSV and save compressed .npz for each split."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw CSV not found at {RAW_CSV}")

    logger.info(f"Reading raw FER2013 CSV from {RAW_CSV}...")
    start_t = time.time()

    data_by_split: dict[str, dict[str, list]] = {
        "train": {"images": [], "labels": [], "record_ids": [], "emotions": []},
        "val": {"images": [], "labels": [], "record_ids": [], "emotions": []},
        "test": {"images": [], "labels": [], "record_ids": [], "emotions": []},
    }

    with open(RAW_CSV, mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            raw_usage = row.get("Usage", row.get("split", "Training")).strip().lower()
            split_key = SPLIT_MAP.get(raw_usage, "train")

            emotion_label = int(row.get("emotion", row.get("label", 0)))
            pixels_str = row.get("pixels", "").strip()
            pixels = [int(p) for p in pixels_str.split()]
            img_arr = np.array(pixels, dtype=np.uint8).reshape((48, 48))

            data_by_split[split_key]["images"].append(img_arr)
            data_by_split[split_key]["labels"].append(emotion_label)
            data_by_split[split_key]["record_ids"].append(idx)
            data_by_split[split_key]["emotions"].append(EMOTION_MAP.get(emotion_label, "unknown"))

    elapsed = time.time() - start_t
    logger.info(f"Parsed all CSV rows in {elapsed:.2f}s.")

    for split, data in data_by_split.items():
        images = np.stack(data["images"], axis=0)  # Shape: (N, 48, 48) uint8
        labels = np.array(data["labels"], dtype=np.int64)  # Shape: (N,) int64
        record_ids = np.array(data["record_ids"], dtype=np.int64)
        emotions = np.array(data["emotions"], dtype=object)

        out_path = OUT_DIR / f"fer2013_{split}.npz"
        np.savez_compressed(
            out_path,
            images=images,
            labels=labels,
            record_ids=record_ids,
            emotions=emotions,
        )
        logger.info(
            f"Saved {split} split to {out_path} | Images shape: {images.shape}, Labels: {labels.shape}"
        )

    logger.info("Dataset caching completed successfully!")


if __name__ == "__main__":
    cache_fer2013()
