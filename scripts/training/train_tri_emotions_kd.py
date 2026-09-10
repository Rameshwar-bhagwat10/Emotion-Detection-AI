"""Distillation-guided fine-tuning for Fear, Sad, and Angry enhancement with frozen early layers."""

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
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

torch.set_num_threads(14)

from ml.models.factory import create_model
from ml.preprocessing.tensor_pipeline import TensorDataStore, IMAGENET_MEAN, IMAGENET_STD

MODELS_DIR = ROOT_DIR / "models" / "v2_optimized"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT_DIR / "reports" / "tri_emotions"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class BalancedTriEmotionPipeline:
    """Balanced mini-batch provider combining FER2013 and curated samples."""

    def __init__(self) -> None:
        fer_train = TensorDataStore("train")
        fer_imgs = fer_train.images.clone()  # [28709, 1, 48, 48]
        fer_lbls = fer_train.labels.clone()  # [28709]

        c_path = ROOT_DIR / "data" / "processed" / "tri_emotions_curated.npz"
        if c_path.exists():
            c_data = np.load(c_path)
            c_raw = c_data["images"]  # [N, 48, 48] uint8
            c_imgs = torch.from_numpy(c_raw).unsqueeze(1).float() / 255.0  # [N, 1, 48, 48]
            c_lbls = torch.from_numpy(c_data["labels"]).long()
        else:
            c_imgs = torch.empty((0, 1, 48, 48), dtype=torch.float32)
            c_lbls = torch.empty((0,), dtype=torch.long)

        self.images = torch.cat([fer_imgs, c_imgs], dim=0).contiguous()
        self.labels = torch.cat([fer_lbls, c_lbls], dim=0).contiguous()
        self.num_samples = len(self.images)
        print(f"BalancedTriEmotionPipeline loaded {self.num_samples} total images.")

    def get_batches(self, batch_size: int = 128, shuffle: bool = True):
        indices = torch.randperm(self.num_samples) if shuffle else torch.arange(self.num_samples)
        for i in range(0, self.num_samples, batch_size):
            b_idx = indices[i : i + batch_size]
            yield self.images[b_idx], self.labels[b_idx]


def evaluate_split(model: nn.Module, split_store: TensorDataStore, device: torch.device, batch_size: int = 128) -> dict:
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
        "angry_f1": float(rep["angry"]["f1-score"]),
        "fear_f1": float(rep["fear"]["f1-score"]),
        "sad_f1": float(rep["sad"]["f1-score"]),
        "happy_f1": float(rep["happy"]["f1-score"]),
        "surprise_f1": float(rep["surprise"]["f1-score"]),
        "neutral_f1": float(rep["neutral"]["f1-score"]),
        "disgust_f1": float(rep["disgust"]["f1-score"]),
        "classification_report": rep,
    }


def train_tri_emotions_kd(epochs: int = 5, lr: float = 2e-5, batch_size: int = 128, kd_alpha: float = 0.5, kd_temp: float = 2.0) -> dict:
    device = torch.device("cpu")
    pipeline = BalancedTriEmotionPipeline()
    val_store = TensorDataStore("val")
    test_store = TensorDataStore("test")

    # Load Teacher Model V2 (Frozen)
    teacher = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    v2_weights = ROOT_DIR / "models" / "v2" / "model.pt"
    s_v2 = torch.load(v2_weights, map_location=device)
    teacher.load_state_dict(s_v2["model_state_dict"] if "model_state_dict" in s_v2 else s_v2)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Load Student Model (Initialized from V2)
    student = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    student.load_state_dict(s_v2["model_state_dict"] if "model_state_dict" in s_v2 else s_v2)
    student.to(device)

    # Freeze early layers (conv1, bn1, layer1, layer2) to prevent catastrophic forgetting
    for name, param in student.named_parameters():
        if any(prefix in name for prefix in ["conv1", "bn1", "layer1", "layer2"]):
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    print(f"Trainable parameters in Student Model: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Baseline evaluation
    base_val = evaluate_split(student, val_store, device)
    print(
        f"Initial Base Val -> Acc: {base_val['accuracy']*100:.2f}% | BalAcc: {base_val['balanced_accuracy']*100:.2f}% | "
        f"MacroF1: {base_val['macro_f1']*100:.2f}% | Angry: {base_val['angry_f1']*100:.2f}% | "
        f"Fear: {base_val['fear_f1']*100:.2f}% | Sad: {base_val['sad_f1']*100:.2f}%"
    )

    best_composite_score = base_val["macro_f1"] + base_val["accuracy"]
    best_student_state = copy.deepcopy(student.state_dict())
    best_epoch = 0
    best_val_metrics = base_val

    mean_dev = IMAGENET_MEAN.to(device)
    std_dev = IMAGENET_STD.to(device)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        student.train()
        total_loss = 0.0
        n_batches = 0

        for b_imgs, b_lbls in pipeline.get_batches(batch_size=batch_size, shuffle=True):
            b_imgs = b_imgs.to(device)
            b_lbls = b_lbls.to(device)

            # Random horizontal flip
            flip_mask = (torch.rand(len(b_imgs), 1, 1, 1, device=device) < 0.5)
            b_imgs = torch.where(flip_mask, torch.flip(b_imgs, dims=[-1]), b_imgs)

            b_112 = F.interpolate(b_imgs, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            b_norm = (b_112 - mean_dev) / std_dev

            optimizer.zero_grad()
            student_logits = student(b_norm)

            with torch.no_grad():
                teacher_logits = teacher(b_norm)

            # 1. Hard Cross Entropy Loss
            ce_loss = F.cross_entropy(student_logits, b_lbls)

            # 2. Knowledge Distillation KL-Divergence Loss
            p_student_soft = F.log_softmax(student_logits / kd_temp, dim=-1)
            p_teacher_soft = F.softmax(teacher_logits / kd_temp, dim=-1)
            kd_loss = F.kl_div(p_student_soft, p_teacher_soft, reduction="batchmean") * (kd_temp ** 2)

            loss = (1.0 - kd_alpha) * ce_loss + kd_alpha * kd_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)
        epoch_sec = time.time() - t0

        # Val Evaluation
        val_res = evaluate_split(student, val_store, device)
        composite_score = val_res["macro_f1"] + val_res["accuracy"]

        is_best = composite_score > best_composite_score
        if is_best:
            best_composite_score = composite_score
            best_student_state = copy.deepcopy(student.state_dict())
            best_epoch = epoch
            best_val_metrics = val_res

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_sec:.1f}s) | Loss: {avg_train_loss:.4f} | "
            f"ValAcc: {val_res['accuracy']*100:.2f}% | BalAcc: {val_res['balanced_accuracy']*100:.2f}% | "
            f"MacroF1: {val_res['macro_f1']*100:.2f}% | Angry: {val_res['angry_f1']*100:.2f}% | "
            f"Fear: {val_res['fear_f1']*100:.2f}% | Sad: {val_res['sad_f1']*100:.2f}% "
            f"{'(* BEST *)' if is_best else ''}"
        )

    # Save Best Model
    torch.save(best_student_state, MODELS_DIR / "model.pt")
    print(f"\nSaved Best Distillation Model to {MODELS_DIR / 'model.pt'} (from Epoch {best_epoch})")

    # Evaluate on Test Set
    best_student = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    best_student.load_state_dict(best_student_state)
    best_student.eval()

    test_res = evaluate_split(best_student, test_store, device)
    print(
        f"\nFinal Test Performance -> Acc: {test_res['accuracy']*100:.2f}% | BalAcc: {test_res['balanced_accuracy']*100:.2f}% | "
        f"MacroF1: {test_res['macro_f1']*100:.2f}% | Angry F1: {test_res['angry_f1']*100:.2f}% | "
        f"Fear F1: {test_res['fear_f1']*100:.2f}% | Sad F1: {test_res['sad_f1']*100:.2f}%"
    )

    return test_res


def main():
    train_tri_emotions_kd(epochs=5, lr=2e-5, batch_size=128, kd_alpha=0.5, kd_temp=2.0)


if __name__ == "__main__":
    main()
