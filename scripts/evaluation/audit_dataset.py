"""Dataset audit and suspicious sample analyzer for Model V2."""

from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "reports" / "v2"

EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def audit_dataset() -> dict[str, dict[str, int]]:
    """Run thorough data audit across train, val, and test splits."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    split_data = {}
    hashes_by_split: dict[str, dict[bytes, list[int]]] = {s: {} for s in splits}

    stats = {}
    for s in splits:
        npz_file = PROCESSED_DIR / f"fer2013_{s}.npz"
        data = np.load(npz_file, allow_pickle=True)
        images = data["images"]
        labels = data["labels"]
        record_ids = data["record_ids"]

        counts = Counter(labels)
        stats[s] = {EMOTION_NAMES[c]: counts[c] for c in range(7)}
        stats[s]["total"] = len(labels)
        split_data[s] = {"images": images, "labels": labels, "record_ids": record_ids}

        for idx, img in enumerate(images):
            h = img.tobytes()
            if h not in hashes_by_split[s]:
                hashes_by_split[s][h] = []
            hashes_by_split[s][h].append(idx)

    # Analyze cross-split duplicates
    train_hashes = set(hashes_by_split["train"].keys())
    val_hashes = set(hashes_by_split["val"].keys())
    test_hashes = set(hashes_by_split["test"].keys())

    train_val_overlap = len(train_hashes.intersection(val_hashes))
    train_test_overlap = len(train_hashes.intersection(test_hashes))
    val_test_overlap = len(val_hashes.intersection(test_hashes))

    logger.info("=== Dataset Split Distribution ===")
    for s, counts in stats.items():
        logger.info(f"Split {s}: {counts}")

    logger.info("=== Duplicate Analysis ===")
    logger.info(f"Train-Val exact duplicates: {train_val_overlap}")
    logger.info(f"Train-Test exact duplicates: {train_test_overlap}")
    logger.info(f"Val-Test exact duplicates: {val_test_overlap}")

    # Generate suspicious samples CSV (e.g. flat/blank images, extreme standard deviation)
    suspicious = []
    for s in splits:
        images = split_data[s]["images"]
        labels = split_data[s]["labels"]
        record_ids = split_data[s]["record_ids"]

        for idx, (img, label, rid) in enumerate(zip(images, labels, record_ids)):
            std = float(np.std(img))
            mean = float(np.mean(img))

            if std < 5.0:
                suspicious.append(
                    {
                        "split": s,
                        "sample_id": int(rid),
                        "label": EMOTION_NAMES[label],
                        "reason": f"Near-zero contrast / flat image (std={std:.2f}, mean={mean:.2f})",
                        "model_prediction": "N/A",
                        "confidence": 0.0,
                    }
                )
            elif mean < 2.0 or mean > 253.0:
                suspicious.append(
                    {
                        "split": s,
                        "sample_id": int(rid),
                        "label": EMOTION_NAMES[label],
                        "reason": f"Extreme saturation / solid tone (mean={mean:.2f})",
                        "model_prediction": "N/A",
                        "confidence": 0.0,
                    }
                )

    csv_path = REPORT_DIR / "suspicious_samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "sample_id", "label", "reason", "model_prediction", "confidence"],
        )
        writer.writeheader()
        for row in suspicious:
            writer.writerow(row)

    logger.info(f"Recorded {len(suspicious)} suspicious samples to {csv_path}")
    return stats


if __name__ == "__main__":
    audit_dataset()
