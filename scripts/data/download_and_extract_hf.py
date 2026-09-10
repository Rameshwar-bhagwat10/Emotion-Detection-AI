"""Download and extract the 433MB facial emotion recognition dataset from Hugging Face."""

import os
from pathlib import Path
import sys
import zipfile
from huggingface_hub import hf_hub_download

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DEST_DIR = DATA_RAW / "kaggle_tapakah68"

def main():
    print("=" * 70)
    print("DOWNLOADING FACIAL EMOTION RECOGNITION DATASET FROM HUGGING FACE")
    print("=" * 70)
    print("Repo: UniqueData/facial-emotion-recognition-dataset")
    print("File: data/images.zip (453 MB)")

    zip_file_path = hf_hub_download(
        repo_id="UniqueData/facial-emotion-recognition-dataset",
        filename="data/images.zip",
        repo_type="dataset",
        local_dir=str(DATA_RAW),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {zip_file_path}")

    # Extract to DEST_DIR
    print(f"Extracting images to {DEST_DIR}...")
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, "r") as z:
        z.extractall(DEST_DIR)

    image_count = len(list(DEST_DIR.rglob("*.jpg")) + list(DEST_DIR.rglob("*.png")) + list(DEST_DIR.rglob("*.jpeg")))
    print(f"Extraction complete! Found {image_count} images in {DEST_DIR}.")

if __name__ == "__main__":
    main()
