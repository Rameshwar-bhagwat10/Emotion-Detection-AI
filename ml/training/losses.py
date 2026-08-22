"""Loss function factory and specialized loss implementations for facial expression recognition.

Includes Focal Loss, Label Smoothing Cross-Entropy, Class-Weighted Cross-Entropy,
and class-weight calculation utilities for addressing severe class imbalance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ml.training.config import LossConfig


class FocalLoss(nn.Module):
    """Multi-class Focal Loss for mitigating extreme class imbalance and hard negative mining.

    Formula:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the ground truth class.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | list[float] | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initialize FocalLoss.

        Args:
            gamma: Focusing parameter (higher = focuses more on hard/misclassified examples).
            alpha: Optional per-class scaling weights tensor of shape [num_classes].
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate multi-class focal loss.

        Args:
            logits: Unnormalized class scores of shape [B, num_classes].
            targets: Ground truth class integer indices of shape [B].

        Returns:
            Computed scalar or per-sample focal loss tensor.
        """
        # Cross entropy loss without reduction yields -log(p_t)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # Probability of ground truth class

        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(
    class_counts: list[int] | np.ndarray | torch.Tensor,
    method: str = "balanced",
    beta: float = 0.9999,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """Compute balanced class weights for addressing severe class imbalance.

    Args:
        class_counts: Array or list of sample counts per class.
        method: 'balanced' (inverse frequency N / (K * N_c)), 'effective_samples' ((1-beta)/(1-beta^N_c)),
                or 'sqrt_inv' (sqrt(N / N_c)).
        beta: Hyperparameter for effective number of samples method.
        max_weight: Upper clipping bound to prevent minority gradients from destabilizing training.

    Returns:
        1D torch.Tensor of normalized class weights.
    """
    counts = np.array(class_counts, dtype=np.float32)
    total = np.sum(counts)
    num_classes = len(counts)

    if method == "balanced":
        # Standard inverse class frequency
        weights = total / (num_classes * np.maximum(counts, 1.0))
    elif method == "effective_samples":
        # Class-Balanced Loss Based on Effective Number of Samples (Cui et al., CVPR 2019)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / np.sum(weights) * num_classes
    elif method == "sqrt_inv":
        weights = np.sqrt(total / np.maximum(counts, 1.0))
        weights = weights / np.sum(weights) * num_classes
    else:
        weights = np.ones(num_classes, dtype=np.float32)

    # Normalize mean to 1.0 and clip max weight
    weights = weights / np.mean(weights)
    weights = np.clip(weights, 0.1, max_weight)

    return torch.tensor(weights, dtype=torch.float32)


def create_loss(
    config: LossConfig | dict[str, Any] | str,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Create and return a configured loss function.

    Args:
        config: LossConfig instance, dictionary, or string name.
        class_weights: Optional 1D tensor of class weights.
        label_smoothing: Label smoothing factor (0.0 to 0.2).
        focal_gamma: Focusing parameter for Focal Loss.

    Returns:
        Configured PyTorch nn.Module loss criterion.
    """
    if isinstance(config, str):
        loss_name = config.lower().strip()
        class_weighted = False
    elif isinstance(config, dict):
        loss_name = str(config.get("name", "cross_entropy")).lower().strip()
        class_weighted = bool(config.get("class_weighted", False))
        label_smoothing = float(config.get("label_smoothing", label_smoothing))
        focal_gamma = float(config.get("focal_gamma", focal_gamma))
    elif isinstance(config, LossConfig):
        loss_name = config.name.lower().strip()
        class_weighted = config.class_weighted
    else:
        raise TypeError(f"Unsupported config type for create_loss: {type(config)}")

    weights = class_weights if class_weighted else None

    if loss_name in ("cross_entropy", "crossentropy", "ce"):
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    elif loss_name in ("focal", "focal_loss", "focalloss"):
        return FocalLoss(gamma=focal_gamma, alpha=weights, reduction="mean")
    elif loss_name in ("label_smoothing", "ls_ce"):
        smoothing = label_smoothing if label_smoothing > 0.0 else 0.1
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)

    raise ValueError(
        f"Unsupported loss function '{loss_name}'. Supported: ['cross_entropy', 'focal', 'label_smoothing']"
    )
