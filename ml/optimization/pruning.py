"""Controlled magnitude pruning, sparsity measurement, and fine-tuning engine."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader

from ml.evaluation.metrics import calculate_macro_metrics, calculate_per_class_metrics


def compute_model_sparsity(model: nn.Module) -> dict[str, Any]:
    """Calculate exact parameter counts, zeroed weights, and sparsity percentage.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary containing total_parameters, zero_parameters, and sparsity_percentage.
    """
    total_weights = 0
    zero_weights = 0

    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            mask = getattr(module, "weight_mask", None)
            if isinstance(mask, torch.Tensor):
                total_weights += mask.numel()
                zero_weights += int(torch.sum(mask == 0).item())
            else:
                w = getattr(module, "weight", None)
                if isinstance(w, torch.Tensor):
                    total_weights += w.numel()
                    zero_weights += int(torch.sum(w == 0).item())

    sparsity_pct = (zero_weights / total_weights * 100.0) if total_weights > 0 else 0.0

    return {
        "total_prunable_parameters": total_weights,
        "zero_parameters": zero_weights,
        "active_parameters": total_weights - zero_weights,
        "sparsity_percentage": round(sparsity_pct, 4),
    }


def apply_magnitude_pruning(
    model: nn.Module,
    amount: float = 0.20,
    method: str = "l1_unstructured",
) -> nn.Module:
    """Apply L1 unstructured magnitude pruning to Conv2D and Linear layers.

    Args:
        model: Base PyTorch neural network.
        amount: Target sparsity fraction in range (0.0, 1.0) (e.g. 0.10, 0.20, 0.30).
        method: Pruning method (default: 'l1_unstructured').

    Returns:
        Pruned PyTorch model with parameter masks applied.
    """
    if not (0.0 < amount < 1.0):
        raise ValueError(f"Pruning amount must be in range (0.0, 1.0), got {amount}")

    pruned_model = copy.deepcopy(model)

    for _, module in pruned_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)

    return pruned_model


def finalize_pruning(model: nn.Module) -> nn.Module:
    """Make pruning permanent by removing reparameterization hooks (required for ONNX export).

    Args:
        model: Pruned PyTorch model.

    Returns:
        Clean PyTorch model with permanently zeroed weights and no pruning hooks.
    """
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, "weight_orig"):
                try:
                    prune.remove(module, "weight")
                except ValueError:
                    pass

    return model


def fine_tune_pruned_model(
    model: nn.Module,
    train_loader: DataLoader[dict[str, Any]],
    val_loader: DataLoader[dict[str, Any]],
    epochs: int = 3,
    learning_rate: float = 0.0001,
    device: torch.device | str = "cpu",
    logger: logging.Logger | None = None,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Fine-tune a pruned model to recover classification accuracy.

    Args:
        model: Pruned PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        epochs: Number of fine-tuning epochs.
        learning_rate: Learning rate for AdamW optimizer.
        device: Hardware device.
        logger: Optional logger instance.

    Returns:
        Tuple of (best fine-tuned model, history dictionary).
    """
    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_macro_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            images = batch["image"].to(dev)
            labels = batch["label"].to(dev)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)

        epoch_train_loss = running_train_loss / max(1, train_samples)

        # Validation step
        model.eval()
        running_val_loss = 0.0
        val_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(dev)
                labels = batch["label"].to(dev)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)

                preds = torch.argmax(outputs, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().tolist())

        epoch_val_loss = running_val_loss / max(1, val_samples)
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        acc = float(np.mean(y_true == y_pred))
        per_class = calculate_per_class_metrics(y_true, y_pred, num_classes=7)
        macro_f1 = calculate_macro_metrics(per_class)["macro_f1"]

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_accuracy"].append(round(acc, 4))
        history["val_macro_f1"].append(round(macro_f1, 4))

        if logger:
            logger.info(
                f"Pruning Fine-tune Epoch {epoch}/{epochs} | Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {acc*100:.2f}% | Val Macro F1: {macro_f1:.4f}"
            )

        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model, history
