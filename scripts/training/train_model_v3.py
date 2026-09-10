"""High-Performance Fast Model V3 Training Engine for Targeted Happy Emotion Recognition."""

from __future__ import annotations

import copy
import json
import math
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

MODELS_DIR = ROOT_DIR / "models" / "v3"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT_DIR / "reports" / "v3"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
HAPPY_IDX = 3


class FastV3TensorPipeline:
    """Pre-loads training data into contiguous memory tensors for vectorized high-speed training."""

    def __init__(self, happy_mix_ratio: float = 0.14) -> None:
        fer_train = TensorDataStore("train")
        fer_imgs = fer_train.images.clone()  # [28709, 1, 48, 48] in [0, 1]
        fer_lbls = fer_train.labels.clone()  # [28709]
        n_fer = len(fer_imgs)

        # Load hard example weights
        weights_file = ROOT_DIR / "data" / "processed" / "fer2013_hard_weights.npz"
        if weights_file.exists():
            w_data = np.load(weights_file)
            fer_weights = torch.from_numpy(w_data["sample_weights"]).float()
        else:
            fer_weights = torch.ones(n_fer, dtype=torch.float32)

        # Load curated Happy dataset
        curated_file = ROOT_DIR / "data" / "processed" / "happy_v3_curated.npz"
        if curated_file.exists():
            cur = np.load(curated_file)
            c_raw = cur["images"]  # [997, 48, 48] uint8
            c_imgs = torch.from_numpy(c_raw).unsqueeze(1).float() / 255.0  # [997, 1, 48, 48]
            c_lbls = torch.full((len(c_imgs),), HAPPY_IDX, dtype=torch.long)
            c_weights = torch.full((len(c_imgs),), 1.25, dtype=torch.float32)
            print(f"Loaded {len(c_imgs)} pristine teacher-verified Happy samples from {curated_file.name}")
        else:
            c_imgs = torch.empty((0, 1, 48, 48), dtype=torch.float32)
            c_lbls = torch.empty((0,), dtype=torch.long)
            c_weights = torch.empty((0,), dtype=torch.float32)

        # Combine
        self.images = torch.cat([fer_imgs, c_imgs], dim=0).contiguous()
        self.labels = torch.cat([fer_lbls, c_lbls], dim=0).contiguous()
        self.weights = torch.cat([fer_weights, c_weights], dim=0).contiguous()
        self.weights = self.weights / self.weights.mean()  # Mean = 1.0

        self.num_samples = len(self.images)
        print(f"FastV3TensorPipeline ready: {self.num_samples} samples in RAM.")

    def get_batches(self, batch_size: int = 128, shuffle: bool = True):
        indices = torch.randperm(self.num_samples) if shuffle else torch.arange(self.num_samples)
        for i in range(0, self.num_samples, batch_size):
            b_idx = indices[i : i + batch_size]
            yield self.images[b_idx], self.labels[b_idx], self.weights[b_idx]


def evaluate_split_fast(model: nn.Module, split_store: TensorDataStore, device: torch.device, batch_size: int = 128) -> dict:
    model.eval()
    all_preds = []
    all_targets = []
    num_samples = split_store.num_samples

    with torch.inference_mode():
        for i in range(0, num_samples, batch_size):
            batch_imgs = split_store.images[i : i + batch_size].to(device)
            batch_lbls = split_store.labels[i : i + batch_size].numpy()

            # Vectorized resize & normalize
            b_112 = F.interpolate(batch_imgs, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            b_norm = (b_112 - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)

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
    happy_prec = float(rep["happy"]["precision"])
    happy_rec = float(rep["happy"]["recall"])
    happy_f1 = float(rep["happy"]["f1-score"])

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "happy_precision": happy_prec,
        "happy_recall": happy_rec,
        "happy_f1": happy_f1,
        "classification_report": rep,
    }


def train_model_v3_fast(epochs: int = 6, lr: float = 5e-5, batch_size: int = 128) -> dict:
    print("=" * 60)
    print("FAST MODEL V3 TRAINING: VECTORIZED TENSOR IN-MEMORY PIPELINE")
    print(f"Epochs: {epochs} | LR: {lr} | Batch Size: {batch_size}")
    print("=" * 60)

    device = torch.device("cpu")
    pipeline = FastV3TensorPipeline(happy_mix_ratio=0.14)
    val_store = TensorDataStore("val")
    test_store = TensorDataStore("test")

    # Load baseline Model V2
    model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    v2_weights_path = ROOT_DIR / "models" / "v2" / "model.pt"
    if not v2_weights_path.exists():
        raise FileNotFoundError(f"Baseline Model V2 missing at {v2_weights_path}")

    v2_state = torch.load(v2_weights_path, map_location=device)
    model.load_state_dict(v2_state["model_state_dict"] if "model_state_dict" in v2_state else v2_state)
    model.to(device)

    # Class balance weights
    lbl_counts = np.bincount(pipeline.labels.numpy(), minlength=7)
    total_lbls = len(pipeline.labels)
    class_weights = total_lbls / (7.0 * np.maximum(lbl_counts, 1).astype(np.float32))
    class_weights = class_weights / class_weights.mean()
    class_weight_tensor = torch.from_numpy(class_weights).float().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Initial Validation
    print("\nEvaluating initial baseline Model V2 on Validation Set...")
    t_start = time.time()
    base_val = evaluate_split_fast(model, val_store, device)
    print(
        f"Baseline V2 Val -> Acc: {base_val['accuracy']*100:.2f}% | BalAcc: {base_val['balanced_accuracy']*100:.2f}% | "
        f"MacroF1: {base_val['macro_f1']*100:.2f}% | Happy Prec: {base_val['happy_precision']*100:.2f}% | "
        f"Happy Rec: {base_val['happy_recall']*100:.2f}% | Happy F1: {base_val['happy_f1']*100:.2f}%"
    )

    best_composite_score = base_val["macro_f1"] + base_val["happy_f1"]
    best_model_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_metrics = base_val
    history = []

    mean_dev = IMAGENET_MEAN.to(device)
    std_dev = IMAGENET_STD.to(device)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for b_imgs, b_lbls, b_weights in pipeline.get_batches(batch_size=batch_size, shuffle=True):
            b_imgs = b_imgs.to(device)
            b_lbls = b_lbls.to(device)
            b_weights = b_weights.to(device)

            # Fast Vectorized Augmentations
            # 1. Random horizontal flip (p=0.5)
            flip_mask = (torch.rand(len(b_imgs), 1, 1, 1, device=device) < 0.5)
            b_imgs = torch.where(flip_mask, torch.flip(b_imgs, dims=[-1]), b_imgs)

            # 2. Subtle brightness jitter for Happy samples (preserve muscle landmarks)
            happy_mask = (b_lbls == HAPPY_IDX).view(-1, 1, 1, 1)
            if happy_mask.any():
                scale = 1.0 + (torch.rand_like(b_imgs) * 0.10 - 0.05)
                b_imgs = torch.where(happy_mask, torch.clamp(b_imgs * scale, 0.0, 1.0), b_imgs)

            # 3. Vectorized batch resize to 112x112 and ImageNet normalize
            b_112 = F.interpolate(b_imgs, size=(112, 112), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
            b_norm = (b_112 - mean_dev) / std_dev

            optimizer.zero_grad()
            logits = model(b_norm)

            ce_loss = F.cross_entropy(logits, b_lbls, weight=class_weight_tensor, reduction="none")
            weighted_loss = (ce_loss * b_weights).mean()

            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += weighted_loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)
        epoch_sec = time.time() - t0

        # Fast Validation Check
        val_res = evaluate_split_fast(model, val_store, device)
        composite_score = val_res["macro_f1"] + val_res["happy_f1"]

        is_best = composite_score > best_composite_score
        if is_best:
            best_composite_score = composite_score
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_metrics = val_res

        log_row = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "val_accuracy": round(val_res["accuracy"], 4),
            "val_balanced_acc": round(val_res["balanced_accuracy"], 4),
            "val_macro_f1": round(val_res["macro_f1"], 4),
            "val_happy_prec": round(val_res["happy_precision"], 4),
            "val_happy_rec": round(val_res["happy_recall"], 4),
            "val_happy_f1": round(val_res["happy_f1"], 4),
            "is_best": is_best,
            "epoch_time_sec": round(epoch_sec, 1),
        }
        history.append(log_row)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_sec:.1f}s) | Loss: {avg_train_loss:.4f} | "
            f"ValAcc: {val_res['accuracy']*100:.2f}% | BalAcc: {val_res['balanced_accuracy']*100:.2f}% | "
            f"MacroF1: {val_res['macro_f1']*100:.2f}% | Happy Prec: {val_res['happy_precision']*100:.2f}% | "
            f"Happy Rec: {val_res['happy_recall']*100:.2f}% | Happy F1: {val_res['happy_f1']*100:.2f}% "
            f"{'(* BEST *)' if is_best else ''}"
        )

    # Save History CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(REPORTS_DIR / "training_history.csv", index=False)

    # Save best Model V3 checkpoint
    torch.save(best_model_state, MODELS_DIR / "model.pt")
    print(f"\nSaved Best Model V3 Checkpoint to {MODELS_DIR / 'model.pt'} (from Epoch {best_epoch})")

    # Export ONNX
    best_model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    best_model.load_state_dict(best_model_state)
    best_model.eval()

    dummy_input = torch.randn(1, 3, 112, 112, dtype=torch.float32)
    onnx_path = MODELS_DIR / "model.onnx"
    torch.onnx.export(
        best_model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"Exported Model V3 ONNX to {onnx_path}")

    # Evaluate on Test Set
    print("\nRunning final evaluation on untouched Test Set (3,589 images)...")
    test_res = evaluate_split_fast(best_model, test_store, device)
    print(
        f"Test Performance -> Acc: {test_res['accuracy']*100:.2f}% | BalAcc: {test_res['balanced_accuracy']*100:.2f}% | "
        f"MacroF1: {test_res['macro_f1']*100:.2f}% | Happy Prec: {test_res['happy_precision']*100:.2f}% | "
        f"Happy Rec: {test_res['happy_recall']*100:.2f}% | Happy F1: {test_res['happy_f1']*100:.2f}%"
    )

    metadata = {
        "model_name": "emotion-resnet18-cbam-v3",
        "version": "v3",
        "description": "Targeted Happy emotion optimized ResNet-18 with CBAM attention, curated AffectNet Happy augmentation, and hard-example mining",
        "input_size": [112, 112],
        "input_channels": 3,
        "classes": {str(i): c for i, c in enumerate(CLASS_NAMES)},
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "best_epoch": best_epoch,
        "total_epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "validation_metrics": {
            "accuracy": best_val_metrics["accuracy"],
            "balanced_accuracy": best_val_metrics["balanced_accuracy"],
            "macro_f1": best_val_metrics["macro_f1"],
            "happy_precision": best_val_metrics["happy_precision"],
            "happy_recall": best_val_metrics["happy_recall"],
            "happy_f1": best_val_metrics["happy_f1"],
        },
        "test_metrics": {
            "accuracy": test_res["accuracy"],
            "balanced_accuracy": test_res["balanced_accuracy"],
            "macro_f1": test_res["macro_f1"],
            "weighted_f1": test_res["weighted_f1"],
            "happy_precision": test_res["happy_precision"],
            "happy_recall": test_res["happy_recall"],
            "happy_f1": test_res["happy_f1"],
        },
        "efficiency": {
            "model_size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 2),
            "total_parameters": sum(p.numel() for p in best_model.parameters()),
        },
        "status": "production_candidate",
    }

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    total_training_sec = time.time() - t_start
    print(f"\nComplete Model V3 Training finished in {total_training_sec:.1f} seconds!")
    return metadata


def main():
    train_model_v3_fast(epochs=6, lr=5e-5, batch_size=128)


if __name__ == "__main__":
    main()
