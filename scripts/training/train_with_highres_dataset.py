"""Fine-tune emotion recognition model with high-resolution dataset, KD regularization, and hard-negative grimace/smile separation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, f1_score

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
REPORTS_DIR = ROOT_DIR / "reports" / "highres_training"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class CombinedEmotionDataset(Dataset):
    def __init__(self, imgs: torch.Tensor, lbls: torch.Tensor, sample_weights: torch.Tensor | None = None):
        self.imgs = imgs  # [N, 1, 48, 48] float in [0, 1]
        self.lbls = lbls  # [N] int64
        self.sample_weights = sample_weights if sample_weights is not None else torch.ones(len(lbls), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.lbls)

    def __getitem__(self, idx: int):
        return self.imgs[idx], self.lbls[idx], self.sample_weights[idx]


def build_training_pipeline():
    # 1. FER2013 train set (28,709 images)
    fer_train = TensorDataStore("train")
    fer_imgs = fer_train.images
    fer_lbls = fer_train.labels
    fer_weights = torch.ones(len(fer_lbls), dtype=torch.float32)

    # 2. Tri-emotions curated AffectNet (1,507 images)
    tri_path = ROOT_DIR / "data" / "processed" / "tri_emotions_curated.npz"
    if tri_path.exists():
        tri_data = np.load(tri_path)
        tri_imgs = torch.from_numpy(tri_data["images"]).unsqueeze(1).float() / 255.0
        tri_lbls = torch.from_numpy(tri_data["labels"]).long()
        tri_weights = torch.full((len(tri_lbls),), 2.0, dtype=torch.float32)
    else:
        tri_imgs = torch.empty((0, 1, 48, 48), dtype=torch.float32)
        tri_lbls = torch.empty(0, dtype=torch.int64)
        tri_weights = torch.empty(0, dtype=torch.float32)

    # 3. High-resolution Studio Curated dataset (1,520 images)
    hr_path = ROOT_DIR / "data" / "processed" / "highres_curated.npz"
    hr_data = np.load(hr_path)
    hr_imgs = torch.from_numpy(hr_data["images"]).unsqueeze(1).float() / 255.0
    hr_lbls = torch.from_numpy(hr_data["labels"]).long()
    
    # Weight high-resolution grimace/disgust and angry samples higher (3.0x) to eliminate teeth-grimace confusion
    hr_weights = torch.ones(len(hr_lbls), dtype=torch.float32)
    for i, lbl in enumerate(hr_lbls):
        if lbl.item() in [0, 1]:  # Angry, Disgust (teeth-bearing grimaces)
            hr_weights[i] = 3.5
        elif lbl.item() in [2, 4]:  # Fear, Sad
            hr_weights[i] = 2.5
        else:
            hr_weights[i] = 1.5

    all_imgs = torch.cat([fer_imgs, tri_imgs, hr_imgs], dim=0)
    all_lbls = torch.cat([fer_lbls, tri_lbls, hr_lbls], dim=0)
    all_weights = torch.cat([fer_weights, tri_weights, hr_weights], dim=0)

    print(f"Total Combined Training Images: {len(all_imgs)}")
    print(f"  - FER2013: {len(fer_imgs)}")
    print(f"  - AffectNet Tri-Emotions: {len(tri_imgs)}")
    print(f"  - High-Res Studio Curated: {len(hr_imgs)}")

    return CombinedEmotionDataset(all_imgs, all_lbls, all_weights)


def evaluate(model, data_store: TensorDataStore, device: torch.device):
    model.eval()
    all_preds = []
    all_targets = []
    mean_dev = IMAGENET_MEAN.to(device)
    std_dev = IMAGENET_STD.to(device)

    with torch.inference_mode():
        for i in range(0, data_store.num_samples, 128):
            batch_imgs = data_store.images[i : i + 128].to(device)
            batch_lbls = data_store.labels[i : i + 128].numpy()

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
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    return acc, bal_acc, macro_f1, per_class_f1


def train():
    device = torch.device("cpu")
    print(f"Starting Fine-Tuning on {device} with PyTorch {torch.__version__} (14 CPU threads)")

    # 1. Load Teacher Model (Frozen)
    teacher = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    teacher_state = torch.load(MODELS_DIR / "model.pt", map_location=device)
    teacher.load_state_dict(teacher_state["model_state_dict"] if "model_state_dict" in teacher_state else teacher_state)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 2. Load Student Model (Fine-tuning target)
    student = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    student.load_state_dict(teacher_state["model_state_dict"] if "model_state_dict" in teacher_state else teacher_state)
    student.to(device)

    # Freeze early layers: conv1, bn1, layer1, layer2
    for name, p in student.named_parameters():
        if any(prefix in name for prefix in ["conv1", "bn1", "layer1", "layer2"]):
            p.requires_grad = False
        else:
            p.requires_grad = True

    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Trainable Parameters in Student Model: {trainable_params:,}")

    # Build dataset & loader
    train_dataset = build_training_pipeline()
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    val_store = TensorDataStore("val")
    test_store = TensorDataStore("test")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=1.2e-4,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4, eta_min=1e-5)

    base_val_acc, base_bal_acc, base_macro_f1, _ = evaluate(student, val_store, device)
    print(f"Initial Validation -> Acc: {base_val_acc*100:.2f}% | BalAcc: {base_bal_acc*100:.2f}% | MacroF1: {base_macro_f1*100:.2f}%\n")

    best_macro_f1 = base_macro_f1
    best_state_dict = None
    temperature = 2.0
    alpha_kd = 0.40

    mean_dev = IMAGENET_MEAN.to(device)
    std_dev = IMAGENET_STD.to(device)

    # Class loss weights to emphasize Disgust & Angry separation from Happy
    class_weights = torch.tensor([1.3, 1.6, 1.2, 0.9, 1.2, 1.1, 1.0], dtype=torch.float32, device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    for epoch in range(1, 5):
        student.train()
        total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for b_imgs, b_lbls, b_w in train_loader:
            b_imgs = b_imgs.to(device)
            b_lbls = b_lbls.to(device)
            b_w = b_w.to(device)

            # Random horizontal flip augmentation
            if torch.rand(1).item() > 0.5:
                b_imgs = torch.flip(b_imgs, dims=[-1])

            # Preprocessing: 48x48 -> 112x112, 3 channels, ImageNet normalization
            b_112 = F.interpolate(b_imgs, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            b_norm = (b_112 - mean_dev) / std_dev

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(b_norm)

            student_logits = student(b_norm)

            # 1. Hard Cross-Entropy Loss (sample-weighted)
            raw_ce = ce_loss_fn(student_logits, b_lbls)
            hard_loss = (raw_ce * b_w).mean()

            # 2. Knowledge Distillation Loss (KL Divergence)
            kd_loss = nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
            ) * (temperature ** 2)

            loss = (1.0 - alpha_kd) * hard_loss + alpha_kd * kd_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.5)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        epoch_time = time.time() - t0
        avg_loss = total_loss / num_batches

        val_acc, val_bal_acc, val_macro_f1, val_f1s = evaluate(student, val_store, device)
        is_best = val_macro_f1 > best_macro_f1
        mark = "(* BEST *)" if is_best else ""

        print(
            f"Epoch [{epoch:02d}/04] ({epoch_time:.1f}s) | "
            f"Loss: {avg_loss:.4f} | "
            f"ValAcc: {val_acc*100:.2f}% | "
            f"BalAcc: {val_bal_acc*100:.2f}% | "
            f"MacroF1: {val_macro_f1*100:.2f}% | "
            f"Angry: {val_f1s[0]*100:.2f}% | Disgust: {val_f1s[1]*100:.2f}% | Happy: {val_f1s[3]*100:.2f}% {mark}"
        )

        if is_best:
            best_macro_f1 = val_macro_f1
            best_state_dict = {k: v.cpu().clone() for k, v in student.state_dict().items()}

    # Save best model
    if best_state_dict is not None:
        save_path = MODELS_DIR / "model.pt"
        torch.save({"model_state_dict": best_state_dict}, save_path)
        student.load_state_dict(best_state_dict)
        print(f"\nSaved Best Model to {save_path}")

    # Export clean ONNX
    dummy_input = torch.randn(1, 3, 112, 112, dtype=torch.float32)
    onnx_path = MODELS_DIR / "model.onnx"
    torch.onnx.export(
        student,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=18,
    )
    print(f"Exported ONNX Model to {onnx_path}")

    # Evaluate on full test set (3,589 images)
    test_acc, test_bal_acc, test_macro_f1, test_f1s = evaluate(student, test_store, device)
    print("\n" + "=" * 65)
    print(f"FINAL UNTOUCHED TEST SET PERFORMANCE (3,589 IMAGES):")
    print("=" * 65)
    print(f"Overall Accuracy:  {test_acc*100:.2f}%")
    print(f"Balanced Accuracy: {test_bal_acc*100:.2f}%")
    print(f"Macro F1-Score:    {test_macro_f1*100:.2f}%")
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c.upper():<9}: {test_f1s[i]*100:5.2f}%")

    with open(REPORTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "test_accuracy": test_acc,
            "balanced_accuracy": test_bal_acc,
            "macro_f1": test_macro_f1,
            "per_class_f1": {c: float(test_f1s[i]) for i, c in enumerate(CLASS_NAMES)},
        }, f, indent=2)


if __name__ == "__main__":
    train()
