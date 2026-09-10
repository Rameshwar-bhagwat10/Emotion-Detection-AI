"""Evaluate emotion detection model on user uploaded photo."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from ml.models.factory import create_model
from ml.preprocessing.tensor_pipeline import IMAGENET_MEAN, IMAGENET_STD

torch.set_num_threads(14)

def run_prediction():
    img_path = Path(r"C:\Users\RAMESHWAR\.gemini\antigravity-ide\brain\30f8f558-1cdd-47d4-9754-31cef0de6c48\.user_uploaded\media_1787462148142.jpg")
    pil_img = Image.open(img_path)
    w, h = pil_img.size

    # Crop tightly to face (eyes to chin, cheek to cheek)
    # The face is centered in the upper half of this portrait
    face_crop = pil_img.crop((int(w * 0.16), int(h * 0.08), int(w * 0.84), int(h * 0.62))).convert("L")
    face_48 = face_crop.resize((48, 48), Image.Resampling.BILINEAR)

    device = torch.device("cpu")
    class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    # 1. Evaluate Optimized Model
    opt_model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    opt_model.load_state_dict(torch.load("models/v2_optimized/model.pt", map_location=device))
    opt_model.eval()

    # 2. Evaluate Baseline Model V2
    v2_model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    v2_state = torch.load("models/v2/model.pt", map_location=device)
    v2_model.load_state_dict(v2_state["model_state_dict"] if "model_state_dict" in v2_state else v2_state)
    v2_model.eval()

    t_img = torch.from_numpy(np.array(face_48, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    t_112 = F.interpolate(t_img, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    t_norm = (t_112 - IMAGENET_MEAN) / IMAGENET_STD

    with torch.inference_mode():
        logits_opt = opt_model(t_norm)
        probs_opt = F.softmax(logits_opt, dim=-1).squeeze(0).numpy()

        logits_v2 = v2_model(t_norm)
        probs_v2 = F.softmax(logits_v2, dim=-1).squeeze(0).numpy()

    pred_opt_idx = int(np.argmax(probs_opt))
    pred_opt_name = class_names[pred_opt_idx]
    conf_opt = float(probs_opt[pred_opt_idx])

    pred_v2_idx = int(np.argmax(probs_v2))
    pred_v2_name = class_names[pred_v2_idx]
    conf_v2 = float(probs_v2[pred_v2_idx])

    print("=" * 65)
    print("EMOTION DETECTION INFERENCE RESULT FOR UPLOADED PHOTO")
    print("=" * 65)
    print(f"Primary Predicted Emotion: {pred_opt_name.upper()} ({conf_opt*100:.2f}% Confidence)")
    print(f"Baseline Model V2 Output:  {pred_v2_name.upper()} ({conf_v2*100:.2f}% Confidence)")
    print("-" * 65)
    print("COMPLETE PROBABILITY DISTRIBUTION BREAKDOWN:")
    print("-" * 65)

    df = pd.DataFrame([
        {
            "Emotion": c.upper(),
            "Optimized Model (%)": f"{probs_opt[i]*100:6.2f}%",
            "Baseline Model V2 (%)": f"{probs_v2[i]*100:6.2f}%",
        }
        for i, c in sorted(enumerate(class_names), key=lambda x: probs_opt[x[0]], reverse=True)
    ])
    print(df.to_string(index=False))

    # Save cropped image preview
    save_preview_path = Path("reports/random_and_internet_eval/uploaded_user_face_crop.png")
    face_crop.save(save_preview_path)
    print(f"\nSaved face crop preview to: {save_preview_path}")

if __name__ == "__main__":
    run_prediction()
