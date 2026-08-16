"""Integration test for Phase 03 Preprocessing Pipeline -> Phase 04 Baseline CNN Model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from ml.models.factory import create_model
from ml.preprocessing.dataloaders import build_dataloaders

RAW_CSV_PATH = Path("data/raw/fer2013/fer2013.csv")


def test_phase3_to_phase4_dataloader_integration() -> None:
    """MANDATORY: Verify real Phase 03 DataLoaders produce batches that pass through BaselineCNN."""
    if not RAW_CSV_PATH.exists():
        pytest.skip("Raw FER2013 dataset not available for integration test.")

    # 1. Create model from factory
    model = create_model("baseline_cnn")
    assert isinstance(model, nn.Module)

    # 2. Build DataLoaders from Phase 03
    train_loader, val_loader, test_loader = build_dataloaders(data_path=RAW_CSV_PATH, batch_size=32)

    # 3. Test Train Batch Integration & Gradient Flow with CrossEntropyLoss
    model.train()
    model.zero_grad()
    criterion = nn.CrossEntropyLoss()

    train_batch = next(iter(train_loader))
    train_imgs = train_batch["image"]
    train_lbls = train_batch["label"]

    assert train_imgs.shape == (32, 1, 48, 48)
    assert train_imgs.dtype == torch.float32

    logits = model(train_imgs)
    assert logits.shape == (32, 7)
    assert logits.dtype == torch.float32
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

    # Compute loss & backward to test differentiability on real data
    loss = criterion(logits, train_lbls)
    assert loss.item() > 0.0
    loss.backward()

    # Verify gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None

    # 4. Test Validation Batch Integration (Eval Mode)
    model.eval()
    val_batch = next(iter(val_loader))
    with torch.no_grad():
        val_logits = model(val_batch["image"])
    assert val_logits.shape == (32, 7)

    # 5. Test Test Batch Integration (Eval Mode)
    test_batch = next(iter(test_loader))
    with torch.no_grad():
        test_logits = model(test_batch["image"])
    assert test_logits.shape == (32, 7)
