"""Extract crops using the exact inference preprocessor pipeline for fine-tuning."""

from pathlib import Path
import json
import cv2
from PIL import Image

from ml.inference.engine import EmotionInferenceEngine

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = ROOT_DIR / "data" / "Sample Images"
CROPS_DIR = ROOT_DIR / "data" / "sample_crops_exact"
CROPS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = ROOT_DIR / "data" / "sample_images_exact_labeled.json"

# Load existing true labels map
with open(ROOT_DIR / "data" / "sample_images_labeled.json") as f:
    orig_labels = json.load(f)

# Map by (image_name, face_index)
label_map = {(it["image"], it["face_id"]): it["true_label"] for it in orig_labels}

engine = EmotionInferenceEngine(auto_load=True)
images = sorted(list(SAMPLES_DIR.glob("*.jpg")))

dataset_records = []

for img_path in images:
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    dets = engine.detector.detect(img_rgb)
    valid_crops, valid_dets = engine._extract_crops(img_rgb, dets)

    for i, (crop_rgb, det) in enumerate(zip(valid_crops, valid_dets)):
        crop_name = f"{img_path.stem}_crop_{i}.jpg"
        crop_path = CROPS_DIR / crop_name
        
        # Save as RGB using PIL
        Image.fromarray(crop_rgb).save(crop_path, quality=95)

        true_label = label_map.get((img_path.name, i), "happy")
        dataset_records.append({
            "image": img_path.name,
            "face_index": i,
            "face_id": det.face_id,
            "crop_file": crop_name,
            "true_label": true_label,
            "bbox": det.bbox.to_tuple(),
        })

with open(OUTPUT_JSON, "w") as f:
    json.dump(dataset_records, f, indent=2)

print(f"Extracted {len(dataset_records)} exact inference crops to {CROPS_DIR}.")
