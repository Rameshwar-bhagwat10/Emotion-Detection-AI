"""Knowledge Distillation engine transferring representations from Champion Teacher to Student."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ml.evaluation.metrics import calculate_macro_metrics, calculate_per_class_metrics
from ml.optimization.config import DistillationConfig


class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining hard-label CrossEntropy with soft-target KL Divergence."""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5) -> None:
        """Initialize Distillation Loss.

        Args:
            temperature: Softmax temperature scaling factor (T > 0).
            alpha: Weight for hard-label CrossEntropy loss (alpha in [0.0, 1.0]).
        """
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in range [0.0, 1.0], got {alpha}")

        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined distillation loss.

        Args:
            student_logits: Unnormalized logits from Student [B, num_classes].
            teacher_logits: Unnormalized logits from Teacher [B, num_classes].
            targets: Ground truth class index tensor [B].

        Returns:
            Tuple of (total_loss, hard_loss, soft_loss).
        """
        hard_loss = self.ce_loss(student_logits, targets)

        # Soft target loss with temperature scaling
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature**2)

        total_loss = (self.alpha * hard_loss) + ((1.0 - self.alpha) * soft_loss)
        return total_loss, hard_loss, soft_loss


def train_student_distillation(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: DataLoader[dict[str, Any]],
    val_loader: DataLoader[dict[str, Any]],
    config: DistillationConfig | None = None,
    device: torch.device | str = "cpu",
    logger: logging.Logger | None = None,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Train Student model guided by frozen Teacher representations.

    Args:
        teacher: Pretrained Champion Teacher model.
        student: Lightweight Student model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Optional DistillationConfig.
        device: Hardware device.
        logger: Optional logger instance.

    Returns:
        Tuple of (trained Student model, history dictionary).
    """
    cfg = config or DistillationConfig()
    dev = torch.device(device) if isinstance(device, str) else device

    # Freeze Teacher completely
    teacher = teacher.to(dev)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Student setup
    student = student.to(dev)
    student.train()
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=1e-4,
    )
    distill_criterion = DistillationLoss(temperature=cfg.temperature, alpha=cfg.alpha)

    best_val_macro_f1 = -1.0
    best_student_weights = copy.deepcopy(student.state_dict())
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        running_train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            images = batch["image"].to(dev)
            labels = batch["label"].to(dev)

            # Obtain Teacher soft labels under no_grad
            with torch.no_grad():
                teacher_logits = teacher(images)

            optimizer.zero_grad()
            student_logits = student(images)

            loss, _, _ = distill_criterion(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)

        epoch_train_loss = running_train_loss / max(1, train_samples)

        # Validation step
        student.eval()
        running_val_loss = 0.0
        val_samples = 0
        all_preds = []
        all_targets = []
        eval_criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(dev)
                labels = batch["label"].to(dev)
                outputs = student(images)
                loss = eval_criterion(outputs, labels)

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
                f"Distillation Epoch {epoch}/{cfg.epochs} | Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {acc*100:.2f}% | Val Macro F1: {macro_f1:.4f}"
            )

        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            best_student_weights = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_student_weights)
    return student, history
