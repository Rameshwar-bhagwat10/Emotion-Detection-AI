"""Evaluate emotion detection model on second user uploaded photo."""

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
    img_path = Path(r"C:\Users\RAMESHWAR\.gemini\antigravity-ide\brain\30f8f558-1cdd-47d4-9754-31cef0de6c48\.user_uploaded\media_1787462376252.jpg")
    pil_img = Image.open(img_path)
    w, h = pil_img.size

    # The face is in the center-top region of the landscape/portrait
    # x: 0.40 to 0.65, y: 0.20 to 0.68
    face_crop = pil_img.crop((int(w * 0.42), int(h * 0.22), int(w * 0.62), int(h * 0.68))).convert("L")
    face_48 = face_crop.resize((48, 48), Image.Resampling.BILINEAR)

    device = torch.device("cpu")
    class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    # Load Optimized Model
    opt_model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    opt_model.load_state_dict(torch.load("models/v2_optimized/model.pt", map_location=device))
    opt_model.eval()

    t_img = torch.from_numpy(np.array(face_48, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    t_112 = F.interpolate(t_img, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    t_norm = (t_112 - IMAGENET_MEAN) / IMAGENET_STD

    with torch.inference_mode():
        logits_opt = opt_model(t_norm)
        probs_opt = F.softmax(logits_opt, dim=-1).squeeze(0).numpy()

    pred_opt_idx = int(np.argmax(probs_opt))
    pred_opt_name = class_names[pred_opt_idx]
    conf_opt = float(probs_opt[pred_opt_idx])

    print("=" * 65)
    print("EMOTION DETECTION INFERENCE RESULT FOR SECOND UPLOADED PHOTO")
    print("=" * 65)
    print(f"Primary Predicted Emotion: {pred_opt_name.upper()} ({conf_opt*100:.2f}% Confidence)")
    print("-" * 65)
    print("COMPLETE PROBABILITY DISTRIBUTION BREAKDOWN:")
    print("-" * 65)

    df = pd.DataFrame([
        {
            "Emotion": c.upper(),
            "Optimized Model (%)": f"{probs_opt[i]*100:6.2f}%",
        }
        for i, c in sorted(enumerate(class_names), key=lambda x: probs_opt[x[0]], reverse=True)
    ])
    print(df.to_string(index=False))

    # Save cropped image preview
    save_preview_path = Path("reports/random_and_internet_eval/uploaded_user_face_crop_2.png")
    face_crop.save(save_preview_path)
    print(f"\nSaved face crop preview to: {save_preview_path}")

if __name__ == "__main__":
    run_prediction()
