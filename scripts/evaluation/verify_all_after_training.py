"""Comprehensive post-training verification across User Uploaded Photos, Internet Images, and Dataset Test Samples."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from ml.models.factory import create_model
from ml.inference.face_detector import create_face_detector
from ml.preprocessing.tensor_pipeline import TensorDataStore, IMAGENET_MEAN, IMAGENET_STD

torch.set_num_threads(14)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def load_model():
    device = torch.device("cpu")
    model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    state = torch.load("models/v2_optimized/model.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    return model, device

def predict_crop_48(model, device, arr_48: np.ndarray):
    t_img = torch.from_numpy(arr_48).unsqueeze(0).unsqueeze(0).float() / 255.0
    t_112 = F.interpolate(t_img, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    t_norm = (t_112 - IMAGENET_MEAN) / IMAGENET_STD
    with torch.inference_mode():
        logits = model(t_norm)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), {CLASS_NAMES[k]: round(float(probs[k]) * 100, 2) for k in range(7)}

def main():
    model, device = load_model()

    print("=" * 75)
    print("VERIFICATION 1: USER UPLOADED IMAGES EVALUATION")
    print("=" * 75)

    user_images = [
        ("Photo 1 (Somber / Serious)", Path("reports/random_and_internet_eval/uploaded_user_face_crop.png")),
        ("Photo 2 (Broad Smile)", Path("reports/random_and_internet_eval/uploaded_user_face_crop_2.png")),
        ("Photo 3 (Wide-Eyed Alert)", Path("reports/random_and_internet_eval/uploaded_user_face_crop_3.png")),
        ("Photo 4 (Teeth-Bearing Grimace)", Path("reports/random_and_internet_eval/uploaded_user_face_crop_4.png")),
    ]

    user_rows = []
    for label, p in user_images:
        if not p.exists():
            continue
        crop_48 = np.array(Image.open(p).convert("L").resize((48, 48)), dtype=np.uint8)
        pred, conf, prob_dict = predict_crop_48(model, device, crop_48)
        top3 = ", ".join([f"{k}: {v}%" for k, v in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]])
        user_rows.append({
            "User Photo": label,
            "Predicted Emotion": pred.upper(),
            "Confidence": f"{conf*100:.1f}%",
            "Top Probabilities": top3,
        })

    user_df = pd.DataFrame(user_rows)
    print(user_df.to_string(index=False))

    print("\n" + "=" * 75)
    print("VERIFICATION 2: 4 REAL-WORLD INTERNET IMAGES")
    print("=" * 75)

    internet_crops = [
        ("Internet #1 (Joyful Smile)", "HAPPY", Path("reports/random_eval_run/internet_1_happy_face_crop.png")),
        ("Internet #2 (Surprise)", "SURPRISE", Path("reports/random_eval_run/internet_2_surprise_face_crop.png")),
        ("Internet #3 (Somber Sad)", "SAD", Path("reports/random_eval_run/internet_3_sad_face_crop.png")),
        ("Internet #4 (Intense Glare)", "ANGRY", Path("reports/random_eval_run/internet_4_angry_face_crop.png")),
    ]

    net_rows = []
    net_correct = 0
    for title, expected, p in internet_crops:
        if not p.exists():
            continue
        crop_48 = np.array(Image.open(p).convert("L").resize((48, 48)), dtype=np.uint8)
        pred, conf, prob_dict = predict_crop_48(model, device, crop_48)
        is_correct = (pred.upper() == expected)
        if is_correct:
            net_correct += 1
        top3 = ", ".join([f"{k}: {v}%" for k, v in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]])
        net_rows.append({
            "Image": title,
            "Expected": expected,
            "Predicted": pred.upper(),
            "Confidence": f"{conf*100:.1f}%",
            "Result": "CORRECT" if is_correct else "INCORRECT",
            "Top Probabilities": top3,
        })

    net_df = pd.DataFrame(net_rows)
    print(net_df.to_string(index=False))

    print("\n" + "=" * 75)
    print("VERIFICATION 3: 4 RANDOM DATASET TEST SAMPLES")
    print("=" * 75)

    test_crops = [
        ("Dataset #1", "HAPPY", Path("reports/random_eval_run/dataset_random_1_happy.png")),
        ("Dataset #2", "SAD", Path("reports/random_eval_run/dataset_random_2_sad.png")),
        ("Dataset #3", "SURPRISE", Path("reports/random_eval_run/dataset_random_3_surprise.png")),
        ("Dataset #4", "ANGRY", Path("reports/random_eval_run/dataset_random_4_angry.png")),
    ]

    ds_rows = []
    ds_correct = 0
    for title, expected, p in test_crops:
        if not p.exists():
            continue
        crop_48 = np.array(Image.open(p).convert("L").resize((48, 48)), dtype=np.uint8)
        pred, conf, prob_dict = predict_crop_48(model, device, crop_48)
        is_correct = (pred.upper() == expected)
        if is_correct:
            ds_correct += 1
        top3 = ", ".join([f"{k}: {v}%" for k, v in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]])
        ds_rows.append({
            "Sample": title,
            "True Emotion": expected,
            "Predicted": pred.upper(),
            "Confidence": f"{conf*100:.1f}%",
            "Result": "CORRECT" if is_correct else "INCORRECT",
            "Top Probabilities": top3,
        })

    ds_df = pd.DataFrame(ds_rows)
    print(ds_df.to_string(index=False))
    print(f"\nDataset Accuracy: {ds_correct}/{len(ds_rows)} ({ds_correct/len(ds_rows)*100:.1f}%)")

if __name__ == "__main__":
    main()
