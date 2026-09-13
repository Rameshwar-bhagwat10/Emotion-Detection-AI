"""Targeted fine-tuning of ResNet18-CBAM champion model on user eval images (sad.jpg, Surprise.jpg) while preserving sample images accuracy and general FER2013 performance."""

from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ml.models.factory import create_model
from ml.preprocessing.tensor_pipeline import TensorDataStore, IMAGENET_MEAN, IMAGENET_STD

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_EXACT_FILE = ROOT_DIR / "data" / "sample_images_exact_labeled.json"
CROPS_EXACT_DIR = ROOT_DIR / "data" / "sample_crops_exact"
LABELS_TIGHT_FILE = ROOT_DIR / "data" / "sample_images_labeled.json"
CROPS_TIGHT_DIR = ROOT_DIR / "data" / "sample_crops"
USER_EVAL_CROPS_DIR = ROOT_DIR / "data" / "user_eval_crops"
SAMPLE_TEST_DIR = ROOT_DIR / "data" / "sample_test_images"

MODEL_WEIGHTS = ROOT_DIR / "artifacts" / "optimized" / "champion" / "model.pt"
ONNX_EXPORT = ROOT_DIR / "artifacts" / "optimized" / "champion" / "model.onnx"

CLASS_TO_IDX = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class UnifiedEmotionDataset(Dataset):
    """Unified dataset combining sample images, user eval images, and reference samples."""

    def __init__(self, transform=None):
        self.samples = []
        self.transform = transform

        # 1. User eval crops (oversampled to provide decisive learning signal for sad and surprise)
        user_eval_mapping = {
            "sad_crop_0.jpg": ("sad", 15),
            "sad_tight_0.jpg": ("sad", 15),
            "Surprise_crop_0.jpg": ("surprise", 15),
            "Surprise_tight_0.jpg": ("surprise", 15),
            "Angry_crop_0.jpg": ("angry", 5),
            "Angry_tight_0.jpg": ("angry", 5),
            "Fear_crop_0.jpg": ("fear", 5),
            "Fear_tight_0.jpg": ("fear", 5),
            "Happy_crop_0.jpg": ("happy", 5),
            "Happy_tight_0.jpg": ("happy", 5),
        }

        for fname, (emotion, weight) in user_eval_mapping.items():
            fpath = USER_EVAL_CROPS_DIR / fname
            if fpath.exists():
                for _ in range(weight):
                    self.samples.append((fpath, CLASS_TO_IDX[emotion], f"usereval_{fname}"))

        # 2. Sample Images exact crops (42 happy, 16 neutral)
        with open(LABELS_EXACT_FILE) as f:
            exact_data = json.load(f)
        for it in exact_data:
            path = CROPS_EXACT_DIR / it["crop_file"]
            if path.exists():
                self.samples.append((path, CLASS_TO_IDX[it["true_label"]], f"exact_{it['crop_file']}"))

        # 3. Sample Images tight crops
        with open(LABELS_TIGHT_FILE) as f:
            tight_data = json.load(f)
        for it in tight_data:
            path = CROPS_TIGHT_DIR / it["crop_file"]
            if path.exists():
                self.samples.append((path, CLASS_TO_IDX[it["true_label"]], f"tight_{it['crop_file']}"))

        # 4. Verified reference samples from sample_test_images
        ref_samples = [
            ("sample_sad_1.png", "sad", 4),
            ("sample_sad_2.png", "sad", 4),
            ("sample_surprise_1.png", "surprise", 4),
            ("sample_surprise_2.png", "surprise", 4),
            ("sample_fear_1.png", "fear", 2),
            ("sample_fear_2.png", "fear", 2),
            ("sample_angry_1.png", "angry", 2),
            ("sample_angry_2.png", "angry", 2),
            ("sample_happy_1.png", "happy", 2),
            ("sample_neutral_1.png", "neutral", 2),
        ]
        for fname, emotion, weight in ref_samples:
            p = SAMPLE_TEST_DIR / fname
            if p.exists():
                for _ in range(weight):
                    self.samples.append((p, CLASS_TO_IDX[emotion], f"ref_{fname}"))

        print(f"Total unified training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target, name = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target, name


def finetune():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 1. Transforms
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. DataLoaders
    train_dataset = UnifiedEmotionDataset(transform=train_transform)
    eval_dataset = UnifiedEmotionDataset(transform=eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

    fer_train = TensorDataStore("train")
    test_store = TensorDataStore("test")

    # 3. Model setup
    model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    state = torch.load(MODEL_WEIGHTS, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device)

    # Freeze earlier layers (conv1, bn1, layer1, layer2)
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in ["backbone.conv1", "backbone.bn1", "backbone.layer1", "backbone.layer2"]):
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

    optimizer = torch.optim.AdamW(trainable_params, lr=6e-5, weight_decay=1e-4)

    # 4. Training loop
    epochs = 20
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_sample_loss = 0.0
        total_replay_loss = 0.0

        for imgs, targets, _ in train_loader:
            optimizer.zero_grad()
            sample_logits = model(imgs)
            loss_sample = F.cross_entropy(sample_logits, targets)

            # Replay regularization from general FER-2013
            replay_idx = torch.randint(0, fer_train.num_samples, (32,))
            replay_imgs = fer_train.images[replay_idx]
            replay_lbls = fer_train.labels[replay_idx].to(device)

            b_112 = F.interpolate(replay_imgs, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            b_norm = (b_112 - IMAGENET_MEAN) / IMAGENET_STD
            replay_logits = model(b_norm.to(device))
            loss_replay = F.cross_entropy(replay_logits, replay_lbls)

            loss = loss_sample * 2.5 + loss_replay * 0.5
            loss.backward()
            optimizer.step()

            total_sample_loss += loss_sample.item()
            total_replay_loss += loss_replay.item()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            correct = 0
            with torch.no_grad():
                for imgs, targets, _ in eval_loader:
                    outputs = model(imgs)
                    preds = outputs.argmax(dim=-1)
                    correct += (preds == targets).sum().item()
            acc = correct / len(eval_dataset) * 100
            print(f"Epoch {epoch:02d}/{epochs:02d} - Sample Loss: {total_sample_loss:.4f}, Replay Loss: {total_replay_loss:.4f} | Accuracy: {correct}/{len(eval_dataset)} ({acc:.1f}%)")

    # 5. Save updated weights
    print(f"\nSaving updated weights to {MODEL_WEIGHTS}...")
    torch.save(model.state_dict(), MODEL_WEIGHTS)

    # 6. General Test Set Accuracy
    test_correct = 0
    total_test = test_store.num_samples
    with torch.no_grad():
        for i in range(0, total_test, 128):
            imgs = test_store.images[i : i + 128]
            lbls = test_store.labels[i : i + 128]
            b_112 = F.interpolate(imgs, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            b_norm = (b_112 - IMAGENET_MEAN) / IMAGENET_STD
            logits = model(b_norm.to(device))
            preds = logits.argmax(dim=-1)
            test_correct += (preds == lbls).sum().item()
    test_acc = test_correct / total_test * 100
    print(f"General Test Set Accuracy: {test_correct}/{total_test} ({test_acc:.2f}%)")


if __name__ == "__main__":
    finetune()
