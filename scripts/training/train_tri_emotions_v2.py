"""Targeted Distillation-Guided Fine-Tuning for Angry, Sad, and Fear Emotion Enhancement."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

torch.set_num_threads(10)

from ml.models.factory import create_model
from ml.preprocessing.tensor_pipeline import TensorDataStore, IMAGENET_MEAN, IMAGENET_STD

MODELS_DIR = ROOT_DIR / "models" / "v2_optimized"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHAMPION_DIR = ROOT_DIR / "artifacts" / "optimized" / "champion"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class TriEmotionCombinedDataset(Dataset):
    """Memory-resident combined dataset fusing FER-2013, curated AffectNet tri-emotions, and studio datasets."""

    def __init__(self) -> None:
        # 1. FER2013 train set (28,709 images)
        fer_train = TensorDataStore("train")
        fer_imgs = fer_train.images.clone()  # [28709, 1, 48, 48] float in [0, 1]
        fer_lbls = fer_train.labels.clone()  # [28709]
        n_fer = len(fer_imgs)

        # 2. Hard Sample Weights for FER-2013
        w_path = ROOT_DIR / "data" / "processed" / "fer2013_tri_hard_weights.npz"
        if w_path.exists():
            w_data = np.load(w_path)
            fer_weights = torch.from_numpy(w_data["sample_weights"]).float()
        else:
            fer_weights = torch.ones(n_fer, dtype=torch.float32)

        # 3. AffectNet Curated Tri-Emotion Dataset (1,507 images: Angry, Fear, Sad)
        c_path = ROOT_DIR / "data" / "processed" / "tri_emotions_curated.npz"
        if c_path.exists():
            c_data = np.load(c_path)
            c_raw = c_data["images"]  # [N, 48, 48] uint8
            c_imgs = torch.from_numpy(c_raw).unsqueeze(1).float() / 255.0  # [N, 1, 48, 48]
            c_lbls = torch.from_numpy(c_data["labels"]).long()
            # High weight on curated samples to strongly teach subtle distinctions
            c_weights = torch.full((len(c_lbls),), 3.0, dtype=torch.float32)
            print(f"Loaded {len(c_imgs)} curated AffectNet samples (Angry={sum(c_lbls==0).item()}, Fear={sum(c_lbls==2).item()}, Sad={sum(c_lbls==4).item()})")
        else:
            c_imgs = torch.empty((0, 1, 48, 48), dtype=torch.float32)
            c_lbls = torch.empty((0,), dtype=torch.long)
            c_weights = torch.empty((0,), dtype=torch.float32)

        # 4. Studio Curated High-Res Dataset (1,520 images)
        hr_path = ROOT_DIR / "data" / "processed" / "highres_curated.npz"
        if hr_path.exists():
            hr_data = np.load(hr_path)
            hr_raw = hr_data["images"]
            hr_imgs = torch.from_numpy(hr_raw).unsqueeze(1).float() / 255.0
            hr_lbls = torch.from_numpy(hr_data["labels"]).long()
            hr_weights = torch.ones(len(hr_lbls), dtype=torch.float32)
            for idx, l in enumerate(hr_lbls):
                if l.item() in [0, 2, 4]:  # Angry, Fear, Sad
                    hr_weights[idx] = 2.5
                elif l.item() == 1:       # Disgust
                    hr_weights[idx] = 2.0
                else:
                    hr_weights[idx] = 1.0
            print(f"Loaded {len(hr_imgs)} high-res studio curated samples")
        else:
            hr_imgs = torch.empty((0, 1, 48, 48), dtype=torch.float32)
            hr_lbls = torch.empty((0,), dtype=torch.long)
            hr_weights = torch.empty((0,), dtype=torch.float32)

        # Combine all datasets
        self.images = torch.cat([fer_imgs, c_imgs, hr_imgs], dim=0).contiguous()
        self.labels = torch.cat([fer_lbls, c_lbls, hr_lbls], dim=0).contiguous()
        self.weights = torch.cat([fer_weights, c_weights, hr_weights], dim=0).contiguous()

        # Normalize weights so mean is 1.0
        self.weights = (self.weights / self.weights.mean()).contiguous()
        self.num_samples = len(self.images)
        print(f"Combined Tri-Emotion Dataset ready with {self.num_samples:,} total training samples.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx], self.weights[idx]


def evaluate_split(model: nn.Module, split_store: TensorDataStore, device: torch.device, batch_size: int = 128) -> dict:
    """Evaluate model on a dataset split returning per-emotion metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    num_samples = split_store.num_samples
    mean_dev = IMAGENET_MEAN.to(device)
    std_dev = IMAGENET_STD.to(device)

    with torch.inference_mode():
        for i in range(0, num_samples, batch_size):
            batch_imgs = split_store.images[i : i + batch_size].to(device)
            batch_lbls = split_store.labels[i : i + batch_size].numpy()

            b_112 = F.interpolate(batch_imgs, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            b_norm = (b_112 - mean_dev) / std_dev

            logits = model(b_norm)
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(batch_lbls)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc = float(np.mean(all_preds == all_targets))
    bal_acc = float(balanced_accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))

    rep = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "angry_recall": float(rep["angry"]["recall"]),
        "fear_recall": float(rep["fear"]["recall"]),
        "sad_recall": float(rep["sad"]["recall"]),
        "angry_f1": float(rep["angry"]["f1-score"]),
        "fear_f1": float(rep["fear"]["f1-score"]),
        "sad_f1": float(rep["sad"]["f1-score"]),
        "disgust_f1": float(rep["disgust"]["f1-score"]),
        "happy_f1": float(rep["happy"]["f1-score"]),
        "surprise_f1": float(rep["surprise"]["f1-score"]),
        "neutral_f1": float(rep["neutral"]["f1-score"]),
        "classification_report": rep,
    }


def train_tri_emotions(
    epochs: int = 6,
    lr: float = 1.2e-4,
    batch_size: int = 128,
    kd_alpha: float = 0.35,
    kd_temp: float = 2.0,
) -> dict:
    device = torch.device("cpu")
    print("=" * 70)
    print("TARGETED TRI-EMOTION ENHANCEMENT ENGINE (ANGRY, SAD, FEAR)")
    print("=" * 70)

    dataset = TriEmotionCombinedDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    val_store = TensorDataStore("val")
    test_store = TensorDataStore("test")

    # 1. Load Teacher Model (Frozen) from Champion
    teacher = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    champ_weights = CHAMPION_DIR / "model.pt"
    s_champ = torch.load(champ_weights, map_location=device)
    teacher_state = s_champ["model_state_dict"] if isinstance(s_champ, dict) and "model_state_dict" in s_champ else s_champ
    teacher.load_state_dict(teacher_state)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 2. Load Student Model (Initialized from Champion)
    student = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    student.load_state_dict(teacher_state)
    student.to(device)

    # 3. Freeze Early Feature Extractors (conv1, bn1, layer1, layer2)
    # Train ONLY layer3, layer4, attention (CBAM modules), and fc classifier head
    for name, param in student.named_parameters():
        if any(prefix in name for prefix in ["conv1", "bn1", "layer1", "layer2"]):
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    print(f"Trainable Parameters (layer3, layer4, CBAM, fc): {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Class weights prioritizing Angry, Fear, Sad separation:
    # classes: ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    class_weights = torch.tensor([1.5, 1.0, 1.6, 0.8, 1.5, 1.0, 0.9], dtype=torch.float32, device=device)

    # Initial baseline evaluation
    base_val = evaluate_split(student, val_store, device)
    base_test = evaluate_split(student, test_store, device)
    print("-" * 70)
    print(
        f"INITIAL BASELINE -> ValAcc: {base_val['accuracy']*100:.2f}% | ValBal: {base_val['balanced_accuracy']*100:.2f}% | "
        f"Angry F1: {base_val['angry_f1']*100:.2f}% | Fear F1: {base_val['fear_f1']*100:.2f}% | Sad F1: {base_val['sad_f1']*100:.2f}%"
    )
    print(
        f"INITIAL TEST     -> TestAcc: {base_test['accuracy']*100:.2f}% | TestBal: {base_test['balanced_accuracy']*100:.2f}% | "
        f"Angry R: {base_test['angry_recall']*100:.2f}% | Fear R: {base_test['fear_recall']*100:.2f}% | Sad R: {base_test['sad_recall']*100:.2f}%"
    )
    print("-" * 70)

    best_score = base_val["macro_f1"] + base_val["angry_f1"] + base_val["fear_f1"] + base_val["sad_f1"] + base_val["balanced_accuracy"]
    best_student_state = copy.deepcopy(student.state_dict())
    best_epoch = 0
    best_metrics = base_val

    mean_dev = IMAGENET_MEAN.to(device)
    std_dev = IMAGENET_STD.to(device)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        student.train()
        total_loss = 0.0
        n_batches = 0

        for b_imgs, b_lbls, b_w in loader:
            b_imgs = b_imgs.to(device)
            b_lbls = b_lbls.to(device)
            b_w = b_w.to(device)

            # Random horizontal flip
            flip_mask = (torch.rand(len(b_imgs), 1, 1, 1, device=device) < 0.5)
            b_imgs = torch.where(flip_mask, torch.flip(b_imgs, dims=[-1]), b_imgs)

            b_112 = F.interpolate(b_imgs, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            b_norm = (b_112 - mean_dev) / std_dev

            optimizer.zero_grad()
            student_logits = student(b_norm)

            with torch.no_grad():
                teacher_logits = teacher(b_norm)

            # 1. Weighted Cross Entropy Loss
            ce_raw = F.cross_entropy(student_logits, b_lbls, weight=class_weights, reduction="none")
            ce_loss = (ce_raw * b_w).mean()

            # 2. Knowledge Distillation Loss (smooth soft label guidance)
            p_student_soft = F.log_softmax(student_logits / kd_temp, dim=-1)
            p_teacher_soft = F.softmax(teacher_logits / kd_temp, dim=-1)
            kd_raw = F.kl_div(p_student_soft, p_teacher_soft, reduction="none").sum(dim=-1) * (kd_temp ** 2)
            kd_loss = (kd_raw * b_w).mean()

            loss = (1.0 - kd_alpha) * ce_loss + kd_alpha * kd_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_sec = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)

        val_res = evaluate_split(student, val_store, device)
        # Composite score targeting Angry, Fear, and Sad enhancement
        comp_score = val_res["macro_f1"] + val_res["angry_f1"] + val_res["fear_f1"] + val_res["sad_f1"] + val_res["balanced_accuracy"]

        is_best = comp_score > best_score
        if is_best:
            best_score = comp_score
            best_student_state = copy.deepcopy(student.state_dict())
            best_epoch = epoch
            best_metrics = val_res

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_sec:.1f}s) | Loss: {avg_loss:.4f} | "
            f"ValAcc: {val_res['accuracy']*100:.2f}% | BalAcc: {val_res['balanced_accuracy']*100:.2f}% | "
            f"Angry F1: {val_res['angry_f1']*100:.2f}% | Fear F1: {val_res['fear_f1']*100:.2f}% | Sad F1: {val_res['sad_f1']*100:.2f}% "
            f"{'(* BEST *)' if is_best else ''}"
        )

    print("-" * 70)
    print(f"Training Complete! Selected Best Weights from Epoch {best_epoch} (Score: {best_score:.4f})")

    # Save to models/v2_optimized/model.pt
    torch.save(best_student_state, MODELS_DIR / "model.pt")
    print(f"Saved optimized checkpoint to {MODELS_DIR / 'model.pt'}")

    # Evaluate on full test set
    best_student = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    best_student.load_state_dict(best_student_state)
    best_student.eval()

    final_test = evaluate_split(best_student, test_store, device)
    print("=" * 70)
    print("FINAL TEST SET BENCHMARK AFTER TRI-EMOTION ENHANCEMENT:")
    print("=" * 70)
    print(f"Overall Accuracy:  {final_test['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy: {final_test['balanced_accuracy']*100:.2f}%")
    print(f"Macro F1-Score:    {final_test['macro_f1']*100:.2f}%")
    print("-" * 70)
    print(f"ANGRY    -> Recall: {final_test['angry_recall']*100:.2f}% | F1: {final_test['angry_f1']*100:.2f}%")
    print(f"FEAR     -> Recall: {final_test['fear_recall']*100:.2f}% | F1: {final_test['fear_f1']*100:.2f}%")
    print(f"SAD      -> Recall: {final_test['sad_recall']*100:.2f}% | F1: {final_test['sad_f1']*100:.2f}%")
    print(f"HAPPY    -> Recall: {final_test['classification_report']['happy']['recall']*100:.2f}% | F1: {final_test['happy_f1']*100:.2f}%")
    print(f"DISGUST  -> Recall: {final_test['classification_report']['disgust']['recall']*100:.2f}% | F1: {final_test['disgust_f1']*100:.2f}%")
    print(f"SURPRISE -> Recall: {final_test['classification_report']['surprise']['recall']*100:.2f}% | F1: {final_test['surprise_f1']*100:.2f}%")
    print(f"NEUTRAL  -> Recall: {final_test['classification_report']['neutral']['recall']*100:.2f}% | F1: {final_test['neutral_f1']*100:.2f}%")
    print("=" * 70)

    return final_test


if __name__ == "__main__":
    train_tri_emotions(epochs=6, lr=1.2e-4, batch_size=128, kd_alpha=0.35, kd_temp=2.0)
