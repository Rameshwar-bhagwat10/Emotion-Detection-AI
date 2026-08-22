"""Automated controlled experiment runner for Model V2 Emotion Recognition program."""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure ROOT_DIR is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from torch import nn

from ml.models.transfer_learning.models_v2 import create_v2_model
from ml.preprocessing.augmentations_v2 import cutmix_data, mixup_criterion, mixup_data
from ml.preprocessing.tensor_pipeline import FastTensorDataLoader, TensorDataStore
from ml.training.losses import FocalLoss, compute_class_weights, create_loss

torch.set_num_threads(14)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = ROOT_DIR / "reports" / "v2"
CHECKPOINTS_DIR = ROOT_DIR / "models" / "v2"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_RESULTS_PATH = REPORTS_DIR / "experiment_results.csv"
MD_LOG_PATH = REPORTS_DIR / "experiments.md"

EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# V1 Baseline metrics reference
V1_BASELINE = {
    "experiment_id": "EXP_V1_BASELINE",
    "model": "resnet18_v1",
    "input_size": "48x48",
    "channels": 1,
    "loss": "cross_entropy",
    "sampler": "uniform",
    "augmentation": "none",
    "optimizer": "adamw",
    "learning_rate": 0.0003,
    "epochs": 15,
    "val_accuracy": 0.5829,
    "val_balanced_accuracy": 0.5043,
    "val_macro_f1": 0.5032,
    "test_accuracy": 0.5860,
    "test_balanced_accuracy": 0.4996,
    "test_macro_f1": 0.4957,
    "weighted_f1": 0.5766,
    "angry_f1": 0.4820,
    "disgust_f1": 0.0952,
    "fear_f1": 0.3398,
    "happy_f1": 0.8282,
    "sad_f1": 0.4423,
    "surprise_f1": 0.6920,
    "neutral_f1": 0.5904,
    "inference_latency_ms": 3.49,
    "model_size_mb": 42.65,
    "notes": "Original V1 baseline (pruned 30% champion)",
}


def init_csv() -> None:
    """Initialize experiment_results.csv with headers and V1 baseline row."""
    with open(CSV_RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(V1_BASELINE.keys()))
        writer.writeheader()
        writer.writerow(V1_BASELINE)


def append_result(res: dict[str, Any]) -> None:
    """Append experiment result to CSV."""
    with open(CSV_RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(V1_BASELINE.keys()))
        writer.writerow(res)


def evaluate_model_tensor(
    model: nn.Module,
    dataloader: FastTensorDataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate model on a FastTensorDataLoader and compute classification metrics and latency."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    start_t = time.perf_counter()
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    total_eval_time = (time.perf_counter() - start_t) * 1000.0
    num_samples = len(all_targets)
    latency_per_sample = total_eval_time / max(num_samples, 1)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float((y_true == y_pred).mean())
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prec, rec, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(7)), zero_division=0
    )

    per_class = {}
    for idx, name in enumerate(EMOTION_NAMES):
        per_class[name] = {
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "f1": float(f1_per_class[idx]),
        }

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": np.array(all_probs),
        "latency_ms": latency_per_sample,
    }


def train_single_experiment_tensor(
    train_store: TensorDataStore,
    val_store: TensorDataStore,
    exp_id: str,
    hypothesis: str,
    model_name: str = "resnet18",
    input_size: tuple[int, int] = (48, 48),
    channels: int = 1,
    sampler_type: str = "uniform",
    loss_type: str = "cross_entropy",
    class_weights_type: str | None = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    augmentation: str = "none",
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    optimizer_name: str = "adamw",
    lr: float = 0.0003,
    epochs: int = 2,
    batch_size: int = 128,
    device_str: str = "cpu",
    limit_samples: int | None = 3000,
    notes: str = "",
) -> dict[str, Any]:
    """Train and evaluate a controlled single experiment with microsecond tensor batching."""
    print(f"\n========================================================", flush=True)
    print(f"EXPERIMENT: {exp_id}", flush=True)
    print(f"Hypothesis: {hypothesis}", flush=True)
    print(f"Config: model={model_name}, size={input_size}, channels={channels}, loss={loss_type}, sampler={sampler_type}", flush=True)

    device = torch.device(device_str)
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Sampler weights
    sample_weights = None
    if sampler_type == "balanced_sampler":
        sample_weights = train_store.get_sample_weights(smoothing=1.0)
    elif sampler_type == "smoothed_sampler":
        sample_weights = train_store.get_sample_weights(smoothing=0.5)

    # 2. Data Loaders
    train_loader = FastTensorDataLoader(
        store=train_store,
        batch_size=batch_size,
        shuffle=(sample_weights is None),
        sample_weights=sample_weights,
        input_size=input_size,
        channels=channels,
        augment=(augmentation != "none"),
        aug_level=augmentation if augmentation != "none" else "moderate",
        limit_samples=limit_samples,
    )

    val_loader = FastTensorDataLoader(
        store=val_store,
        batch_size=batch_size,
        shuffle=False,
        input_size=input_size,
        channels=channels,
        augment=False,
        limit_samples=1200,
    )

    # 3. Model
    model = create_v2_model(
        name=model_name,
        num_classes=7,
        pretrained=True,
        in_channels=channels,
        dropout_rate=0.2,
    ).to(device)

    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    # 4. Loss
    class_counts = train_store.get_class_counts()
    weights = None
    if class_weights_type in ("balanced", "effective"):
        method_name = "balanced" if class_weights_type == "balanced" else "effective_samples"
        weights = compute_class_weights(class_counts, method=method_name).to(device)

    if loss_type == "focal":
        criterion = FocalLoss(gamma=focal_gamma, alpha=weights).to(device)
    elif loss_type == "label_smoothing" or label_smoothing > 0.0:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing if label_smoothing > 0.0 else 0.1).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights).to(device)

    # 5. Optimizer & Scheduler
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_macro_f1 = 0.0
    best_val_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_ep = time.time()
        n_batches = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            if mixup_alpha > 0.0:
                imgs_mixed, y_a, y_b, lam = mixup_data(imgs, targets, alpha=mixup_alpha)
                logits = model(imgs_mixed)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            elif cutmix_alpha > 0.0:
                imgs_cut, y_a, y_b, lam = cutmix_data(imgs, targets, alpha=cutmix_alpha)
                logits = model(imgs_cut)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(imgs)
                loss = criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        ep_loss = train_loss / max(n_batches, 1)
        ep_duration = time.time() - start_ep

        val_eval = evaluate_model_tensor(model, val_loader, device)
        val_macro_f1 = val_eval["macro_f1"]
        val_acc = val_eval["accuracy"]

        print(
            f"  Epoch {epoch}/{epochs} ({ep_duration:.1f}s) | Loss: {ep_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_macro_f1*100:.2f}% | "
            f"Disgust F1: {val_eval['per_class']['disgust']['f1']*100:.2f}% | "
            f"Fear F1: {val_eval['per_class']['fear']['f1']*100:.2f}%",
            flush=True,
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_metrics = val_eval

    result_row = {
        "experiment_id": exp_id,
        "model": model_name,
        "input_size": f"{input_size[0]}x{input_size[1]}",
        "channels": channels,
        "loss": loss_type,
        "sampler": sampler_type,
        "augmentation": augmentation,
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "epochs": epochs,
        "val_accuracy": round(best_val_metrics.get("accuracy", 0.0), 4),
        "val_balanced_accuracy": round(best_val_metrics.get("balanced_accuracy", 0.0), 4),
        "val_macro_f1": round(best_val_metrics.get("macro_f1", 0.0), 4),
        "test_accuracy": "HELD_OUT",
        "test_balanced_accuracy": "HELD_OUT",
        "test_macro_f1": "HELD_OUT",
        "weighted_f1": round(best_val_metrics.get("weighted_f1", 0.0), 4),
        "angry_f1": round(best_val_metrics.get("per_class", {}).get("angry", {}).get("f1", 0.0), 4),
        "disgust_f1": round(best_val_metrics.get("per_class", {}).get("disgust", {}).get("f1", 0.0), 4),
        "fear_f1": round(best_val_metrics.get("per_class", {}).get("fear", {}).get("f1", 0.0), 4),
        "happy_f1": round(best_val_metrics.get("per_class", {}).get("happy", {}).get("f1", 0.0), 4),
        "sad_f1": round(best_val_metrics.get("per_class", {}).get("sad", {}).get("f1", 0.0), 4),
        "surprise_f1": round(best_val_metrics.get("per_class", {}).get("surprise", {}).get("f1", 0.0), 4),
        "neutral_f1": round(best_val_metrics.get("per_class", {}).get("neutral", {}).get("f1", 0.0), 4),
        "inference_latency_ms": round(best_val_metrics.get("latency_ms", 0.0), 2),
        "model_size_mb": round(param_size, 2),
        "notes": notes,
    }

    append_result(result_row)
    return {
        "metrics": best_val_metrics,
        "row": result_row,
        "model": model,
    }
