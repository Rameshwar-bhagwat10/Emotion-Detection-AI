"""Download and acquire the FER2013 facial expression dataset.

Downloads verified parquet splits from Hugging Face mirror, converts into canonical
raw CSV format at `data/raw/fer2013/fer2013.csv`, and generates `data/raw/fer2013/metadata.json`.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

# Add root directory to pythonpath
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download-dataset")

CONFIG_PATH = ROOT_DIR / "ml" / "configs" / "dataset.yaml"


def load_config() -> dict[str, Any]:
    """Load dataset configuration."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return dict(data) if isinstance(data, dict) else {}


def download_split_parquet(url: str, split_name: str) -> pd.DataFrame:
    """Download a parquet split file with progress bar."""
    logger.info("Downloading %s split from %s...", split_name, url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req, timeout=60) as resp:
        total_size = int(resp.headers.get("content-length", 0))
        buffer = io.BytesIO()
        chunk_size = 1024 * 64

        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc=f"Split [{split_name}]"
        ) as pbar:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
                pbar.update(len(chunk))

        buffer.seek(0)
        df = pd.read_parquet(buffer)
        logger.info("Successfully downloaded %s split (%d records)", split_name, len(df))
        return df


def convert_image_bytes_to_pixels_str(img_val: Any) -> str:
    """Extract (48, 48) grayscale array from bytes or dict and format as space-delimited string."""
    if isinstance(img_val, dict) and "bytes" in img_val:
        raw_bytes = img_val["bytes"]
    elif isinstance(img_val, bytes):
        raw_bytes = img_val
    elif isinstance(img_val, str):
        return img_val
    else:
        raise ValueError(f"Unsupported image object: {type(img_val)}")

    with Image.open(io.BytesIO(raw_bytes)) as pil_img:
        gray = pil_img.convert("L")
        if gray.size != (48, 48):
            gray = gray.resize((48, 48), Image.Resampling.BILINEAR)
        arr = np.array(gray, dtype=np.uint8).flatten()
        return " ".join(map(str, arr))


def acquire_fer2013(force: bool = False) -> None:
    """Acquire the FER2013 dataset and write canonical raw CSV and metadata manifest."""
    config = load_config()
    ds_conf = config.get("dataset", {})
    paths_conf = ds_conf.get("paths", {})
    urls_conf = ds_conf.get("urls", {})

    raw_dir = ROOT_DIR / paths_conf.get("raw_dir", "data/raw/fer2013")
    csv_file = ROOT_DIR / paths_conf.get("csv_file", "data/raw/fer2013/fer2013.csv")
    metadata_file = ROOT_DIR / paths_conf.get("metadata_file", "data/raw/fer2013/metadata.json")

    raw_dir.mkdir(parents=True, exist_ok=True)

    if csv_file.exists() and not force:
        logger.info("Dataset already exists at '%s'. Use --force to re-download.", csv_file)
        return

    logger.info("Starting FER2013 dataset acquisition...")

    # Map of split url -> usage name
    splits_to_fetch = [
        ("train", urls_conf.get("train_parquet"), "Training"),
        ("val", urls_conf.get("valid_parquet"), "PublicTest"),
        ("test", urls_conf.get("test_parquet"), "PrivateTest"),
    ]

    all_rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}

    for split_key, url, usage_label in splits_to_fetch:
        if not url:
            raise ValueError(f"Missing download URL for split '{split_key}' in dataset.yaml")

        df = download_split_parquet(url, split_key)
        split_counts[split_key] = len(df)

        logger.info("Formatting %d rows for split '%s'...", len(df), usage_label)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing [{split_key}]"):
            label = int(row["label"])
            pixels_str = convert_image_bytes_to_pixels_str(row["image"])
            all_rows.append(
                {
                    "emotion": label,
                    "pixels": pixels_str,
                    "Usage": usage_label,
                }
            )

    # Save to canonical fer2013.csv
    logger.info("Writing %d records to canonical CSV '%s'...", len(all_rows), csv_file)
    combined_df = pd.DataFrame(all_rows)
    combined_df.to_csv(csv_file, index=False)
    logger.info("Wrote %s (Size: %.2f MB)", csv_file, csv_file.stat().st_size / (1024 * 1024))

    # Save metadata.json
    metadata = {
        "dataset_name": ds_conf.get("name", "FER2013"),
        "dataset_version": ds_conf.get("version", "1.0.0"),
        "description": ds_conf.get("description"),
        "source": ds_conf.get("source"),
        "license": ds_conf.get("license"),
        "num_classes": ds_conf.get("image", {}).get("num_classes", 7),
        "classes": list(ds_conf.get("classes", {}).values()),
        "image_width": ds_conf.get("image", {}).get("width", 48),
        "image_height": ds_conf.get("image", {}).get("height", 48),
        "color_mode": ds_conf.get("image", {}).get("color_mode", "grayscale"),
        "total_samples": len(combined_df),
        "splits": {
            "train": split_counts.get("train", 0),
            "val": split_counts.get("val", 0),
            "test": split_counts.get("test", 0),
        },
        "canonical_csv": str(csv_file.relative_to(ROOT_DIR)),
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved dataset metadata to '%s'", metadata_file)
    logger.info("Dataset acquisition completed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FER2013 dataset")
    parser.add_argument(
        "--force", action="store_true", help="Force re-download if file already exists"
    )
    args = parser.parse_args()

    try:
        acquire_fer2013(force=args.force)
    except Exception as exc:
        logger.error("Dataset acquisition failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
