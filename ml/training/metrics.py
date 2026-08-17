"""Training and validation metric calculation utilities."""

from __future__ import annotations

import torch


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate classification accuracy from raw logits and integer labels.

    Args:
        logits: Unnormalized class logits of shape [B, num_classes].
        labels: Ground truth class integer indices of shape [B].

    Returns:
        Accuracy as a float between 0.0 and 1.0.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D [B, C], got shape {tuple(logits.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D [B], got shape {tuple(labels.shape)}")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Batch size mismatch: logits has {logits.shape[0]}, labels has {labels.shape[0]}"
        )

    if logits.shape[0] == 0:
        return 0.0

    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return float(correct / logits.shape[0])


class MetricTracker:
    """Accumulates batch losses and predictions to compute exact sample-weighted epoch metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated counts and metrics."""
        self.total_loss: float = 0.0
        self.total_correct: int = 0
        self.total_samples: int = 0

    def update(
        self,
        loss_val: float,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int | None = None,
    ) -> None:
        """Update tracker with a batch's scalar loss and tensor predictions.

        Args:
            loss_val: The scalar loss value (e.g. from loss.item()).
            logits: Model output raw logits of shape [B, C].
            labels: Ground truth labels of shape [B].
            batch_size: Optional explicit batch size (defaults to logits.shape[0]).
        """
        bs = logits.shape[0] if batch_size is None else batch_size
        if bs <= 0:
            return

        self.total_loss += float(loss_val) * bs
        predictions = logits.argmax(dim=1)
        self.total_correct += int((predictions == labels).sum().item())
        self.total_samples += bs

    def compute(self) -> dict[str, float]:
        """Compute epoch-level sample-weighted loss and accuracy.

        Returns:
            Dictionary containing 'loss' and 'accuracy'.
        """
        if self.total_samples == 0:
            return {"loss": 0.0, "accuracy": 0.0}

        avg_loss = self.total_loss / self.total_samples
        accuracy = self.total_correct / self.total_samples
        return {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
        }
