"""Unit tests for Knowledge Distillation loss and training mechanics."""

from __future__ import annotations

import torch

from ml.optimization.distillation import DistillationLoss


def test_distillation_loss_forward_and_backward() -> None:
    """Verify DistillationLoss computes finite losses and gradients flow to student logits."""
    criterion = DistillationLoss(temperature=4.0, alpha=0.5)

    student_logits = torch.randn(4, 7, requires_grad=True)
    teacher_logits = torch.randn(4, 7)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    total_loss, hard_loss, soft_loss = criterion(student_logits, teacher_logits, targets)

    assert torch.isfinite(total_loss)
    assert torch.isfinite(hard_loss)
    assert torch.isfinite(soft_loss)
    assert total_loss.item() > 0.0

    total_loss.backward()
    assert student_logits.grad is not None
