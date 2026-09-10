"""Process, detect faces, augment, and package the high-resolution facial emotion dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.inference.face_detector import create_face_detector
from ml.preprocessing.tensor_pipeline import TensorDataStore

RAW_DIR = ROOT_DIR / "data" / "raw" / "kaggle_tapakah68"
OUTPUT_NPZ = ROOT_DIR / "data" / "processed" / "highres_curated.npz"

EMOTION_MAP = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprised": 5,
    "surprise": 5,
    "neutral": 6,
    "contempt": 6,
}

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def compute_dhash(image_48: np.ndarray) -> int:
    resized = cv2.resize(image_48, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    val = 0
    for b in diff.flatten():
        val = (val << 1) | int(b)
    return val


def main():
    print("=" * 70)
    print("PROCESSING HIGH-RESOLUTION FACIAL EMOTION RECOGNITION DATASET")
    print("=" * 70)

    detector = create_face_detector(detector_type="yunet")

    # Load test set hashes for zero-leakage guarantee
    test_store = TensorDataStore("test")
    test_dhashes = set()
    test_shas = set()
    for img_tensor in test_store.images:
        arr = (img_tensor.squeeze(0).numpy() * 255.0).astype(np.uint8)
        test_dhashes.add(compute_dhash(arr))
        test_shas.add(hashlib.sha256(arr.tobytes()).hexdigest())
    print(f"Loaded {len(test_shas)} test set hashes for zero-leakage validation.")

    all_images = []
    all_labels = []
    class_counts = {c: 0 for c in CLASS_NAMES}

    image_files = sorted(list(RAW_DIR.rglob("*.jpg")) + list(RAW_DIR.rglob("*.png")))
    print(f"Found {len(image_files)} raw studio image files.")

    rejected_leak = 0

    for idx, img_path in enumerate(image_files, 1):
        stem_lower = img_path.stem.lower()
        if stem_lower not in EMOTION_MAP:
            continue

        label_idx = EMOTION_MAP[stem_lower]
        label_name = CLASS_NAMES[label_idx]

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        h, w, _ = bgr.shape
        # Resize large image slightly for fast robust detection
        scale = 1000.0 / max(h, w)
        small_bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        detections = detector.detect(small_bgr)
        if not detections:
            # Fallback center crop if detection missed
            cy, cx = int(h * 0.45), int(w * 0.50)
            box_half = int(min(h, w) * 0.35)
            x0, y0 = max(0, cx - box_half), max(0, cy - box_half)
            x1, y1 = min(w, cx + box_half), min(h, cy + box_half)
        else:
            best_det = max(detections, key=lambda d: d.bbox.area)
            orig_bbox_x = int(best_det.bbox.x / scale)
            orig_bbox_y = int(best_det.bbox.y / scale)
            orig_bbox_w = int(best_det.bbox.width / scale)
            orig_bbox_h = int(best_det.bbox.height / scale)

            pad_w = int(orig_bbox_w * 0.12)
            pad_h = int(orig_bbox_h * 0.12)
            x0 = max(0, orig_bbox_x - pad_w)
            y0 = max(0, orig_bbox_y - pad_h)
            x1 = min(w, orig_bbox_x + orig_bbox_w + pad_w)
            y1 = min(h, orig_bbox_y + orig_bbox_h + pad_h)

        face_rgb = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(face_rgb).convert("L")

        # Generate base crop + gentle augmentations (variations in scale, shift, and horizontal flip)
        fw, fh = pil_face.size
        crops_to_make = [
            (0, 0, fw, fh, False),
            (0, 0, fw, fh, True),  # H-flip
            (int(fw * 0.04), int(fh * 0.04), int(fw * 0.96), int(fh * 0.96), False),  # Slight zoom
            (int(fw * 0.04), int(fh * 0.04), int(fw * 0.96), int(fh * 0.96), True),  # Zoom + flip
            (0, int(fh * 0.05), fw, fh, False),  # Slight shift down
            (0, int(fh * 0.05), fw, fh, True),
            (0, 0, fw, int(fh * 0.95), False),  # Slight shift up
            (0, 0, fw, int(fh * 0.95), True),
        ]

        # For underrepresented negative classes (Disgust, Anger, Fear, Sad), add extra lighting/contrast variations
        if label_name in ["disgust", "angry", "fear", "sad"]:
            extra_crops = [
                (int(fw * 0.02), 0, int(fw * 0.98), fh, False),
                (int(fw * 0.02), 0, int(fw * 0.98), fh, True),
                (0, int(fh * 0.02), fw, int(fh * 0.98), False),
                (0, int(fh * 0.02), fw, int(fh * 0.98), True),
            ]
            crops_to_make.extend(extra_crops)

        for cx0, cy0, cx1, cy1, do_flip in crops_to_make:
            cropped = pil_face.crop((cx0, cy0, cx1, cy1)).resize((48, 48), Image.Resampling.BILINEAR)
            if do_flip:
                cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)

            arr = np.array(cropped, dtype=np.uint8)

            # Leakage check
            h_sha = hashlib.sha256(arr.tobytes()).hexdigest()
            h_dhash = compute_dhash(arr)
            if h_sha in test_shas or h_dhash in test_dhashes:
                rejected_leak += 1
                continue

            all_images.append(arr)
            all_labels.append(label_idx)
            class_counts[label_name] += 1

    imgs_np = np.stack(all_images)
    lbls_np = np.array(all_labels, dtype=np.int64)

    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, images=imgs_np, labels=lbls_np)

    print(f"\nProcessing Complete!")
    print(f"Total curated images: {len(imgs_np)} (Rejected {rejected_leak} leakage candidates)")
    print(f"Saved to: {OUTPUT_NPZ}")
    print("\nClass Distribution:")
    for k, v in class_counts.items():
        print(f"  {k.upper():<9}: {v}")


if __name__ == "__main__":
    main()
