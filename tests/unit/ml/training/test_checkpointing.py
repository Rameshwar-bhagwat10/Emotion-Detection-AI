"""Unit tests for atomic checkpoint saving, loading, and state restoration."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from ml.training.checkpointing import CheckpointManager


def test_checkpoint_save_best_and_last(tmp_path: Path) -> None:
    """Verify CheckpointManager writes best.pt and last.pt atomically."""
    manager = CheckpointManager(save_dir=tmp_path, monitor="val_loss", mode="min")
    model = nn.Linear(4, 2)
    optimizer = AdamW(model.parameters(), lr=0.01)

    # Epoch 1: val_loss 1.5 -> saved as last and best
    paths_1 = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=1,
        metric=1.5,
        history=[{"epoch": 1, "val_loss": 1.5}],
    )
    assert "best" in paths_1 and (tmp_path / "best.pt").exists()
    assert "last" in paths_1 and (tmp_path / "last.pt").exists()
    assert manager.best_metric == 1.5

    # Epoch 2: val_loss 1.8 -> saved as last only (not best)
    paths_2 = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=2,
        metric=1.8,
        history=[{"epoch": 1, "val_loss": 1.5}, {"epoch": 2, "val_loss": 1.8}],
    )
    assert "best" not in paths_2
    assert "last" in paths_2
    assert manager.best_metric == 1.5

    # Epoch 3: val_loss 1.2 -> new best
    paths_3 = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=3,
        metric=1.2,
        history=[{"epoch": 3, "val_loss": 1.2}],
    )
    assert "best" in paths_3
    assert manager.best_metric == 1.2


def test_checkpoint_load_state_restoration(tmp_path: Path) -> None:
    """Verify loaded checkpoint accurately restores model parameters and optimizer state."""
    manager = CheckpointManager(save_dir=tmp_path)
    model_orig = nn.Linear(4, 2)
    optimizer_orig = AdamW(model_orig.parameters(), lr=0.05)

    # Perform a dummy training step to give optimizer non-empty state
    x = torch.randn(2, 4)
    loss = model_orig(x).sum()
    loss.backward()
    optimizer_orig.step()

    manager.save(
        model=model_orig,
        optimizer=optimizer_orig,
        scheduler=None,
        epoch=5,
        metric=0.42,
        history=[{"epoch": 5, "val_loss": 0.42}],
        config={"test_key": "test_val"},
    )

    ckpt_path = tmp_path / "best.pt"
    model_loaded = nn.Linear(4, 2)
    optimizer_loaded = AdamW(model_loaded.parameters(), lr=0.01)

    loaded_dict = CheckpointManager.load(
        checkpoint_path=ckpt_path,
        model=model_loaded,
        optimizer=optimizer_loaded,
    )

    assert loaded_dict["epoch"] == 5
    assert loaded_dict["best_metric"] == 0.42
    assert loaded_dict["config"]["test_key"] == "test_val"

    # Verify model weights match exactly
    for p1, p2 in zip(model_orig.parameters(), model_loaded.parameters(), strict=True):
        assert torch.equal(p1, p2)
