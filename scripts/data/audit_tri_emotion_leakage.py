"""Cryptographic SHA-256 and Perceptual dHash Leakage Auditor for Curated Tri-Emotion Dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from ml.preprocessing.tensor_pipeline import TensorDataStore

CURATED_DIR = ROOT_DIR / "data" / "interim" / "tri_emotions_curated"
REPORTS_DIR = ROOT_DIR / "reports" / "tri_emotions"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_sha256(img_arr: np.ndarray) -> str:
    return hashlib.sha256(img_arr.tobytes()).hexdigest()


def compute_dhash(img_arr: np.ndarray, hash_size: int = 8) -> int:
    pil_img = Image.fromarray(img_arr).resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(pil_img, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def run_tri_emotion_leakage_audit() -> dict:
    print("=" * 60)
    print("STARTING ZERO-LEAKAGE AUDIT FOR ANGRY, FEAR, AND SAD DATASET")
    print("=" * 60)

    # 1. Load FER2013 splits
    test_store = TensorDataStore("test")
    val_store = TensorDataStore("val")
    train_store = TensorDataStore("train")

    def extract_hashes(store: TensorDataStore):
        imgs = (store.images.squeeze(1).numpy() * 255.0).astype(np.uint8)
        sha_set = set()
        dhash_list = []
        for img in imgs:
            sha_set.add(compute_sha256(img))
            dhash_list.append(compute_dhash(img))
        return sha_set, dhash_list

    print("Hashing FER2013 test, val, and train splits...")
    test_sha, test_dhashes = extract_hashes(test_store)
    val_sha, val_dhashes = extract_hashes(val_store)
    train_sha, train_dhashes = extract_hashes(train_store)

    # 2. Audit Candidate Images
    cand_files = sorted(list(CURATED_DIR.glob("*.png")))
    print(f"Found {len(cand_files)} candidate images in {CURATED_DIR}...")

    accepted = []
    rejected = []
    seen_cand_sha = set()

    for fpath in cand_files:
        img_arr = np.array(Image.open(fpath))
        sha = compute_sha256(img_arr)
        dh = compute_dhash(img_arr)

        # Internal duplicate check
        if sha in seen_cand_sha:
            rejected.append({"file": fpath.name, "reason": "internal_exact_duplicate"})
            fpath.unlink()
            continue
        seen_cand_sha.add(sha)

        # Exact SHA match with test set
        if sha in test_sha:
            rejected.append({"file": fpath.name, "reason": "test_sha_exact_collision"})
            fpath.unlink()
            continue

        # Exact SHA match with val set
        if sha in val_sha:
            rejected.append({"file": fpath.name, "reason": "val_sha_exact_collision"})
            fpath.unlink()
            continue

        # Exact SHA match with train set
        if sha in train_sha:
            rejected.append({"file": fpath.name, "reason": "train_sha_exact_collision"})
            fpath.unlink()
            continue

        # Perceptual near-duplicate check against test set (Hamming <= 4)
        is_perceptual_leak = False
        for t_dh in test_dhashes:
            if hamming_distance(dh, t_dh) <= 4:
                is_perceptual_leak = True
                break

        if is_perceptual_leak:
            rejected.append({"file": fpath.name, "reason": "test_perceptual_collision_hamming_le_4"})
            fpath.unlink()
            continue

        accepted.append(fpath.name)

    audit_summary = {
        "total_evaluated_candidates": len(cand_files),
        "clean_accepted_candidates": len(accepted),
        "total_rejected": len(rejected),
        "test_sha_collisions": sum(1 for r in rejected if r["reason"] == "test_sha_exact_collision"),
        "test_perceptual_collisions": sum(1 for r in rejected if r["reason"] == "test_perceptual_collision_hamming_le_4"),
        "test_leakage_status": "ZERO_LEAKAGE_VERIFIED",
    }

    with open(REPORTS_DIR / "tri_emotion_leakage_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    pd.DataFrame(rejected).to_csv(REPORTS_DIR / "tri_emotion_rejections.csv", index=False)

    print("\nAUDIT SUMMARY:")
    for k, v in audit_summary.items():
        print(f"  {k}: {v}")
    print(f"\nFinal count of verified clean, leak-free samples: {len(accepted)}")
    return audit_summary


def main():
    run_tri_emotion_leakage_audit()


if __name__ == "__main__":
    main()
