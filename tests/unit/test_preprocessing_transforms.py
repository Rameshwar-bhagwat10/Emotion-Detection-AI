"""Unit tests for Phase 03 Transforms & Determinism Verification."""

from __future__ import annotations

import numpy as np
import torch

from ml.preprocessing.transforms import (
    build_test_transform,
    build_train_transform,
    build_val_transform,
)


def test_train_transform_output_shape_and_dtype() -> None:
    """Verify training transform produces [1, 48, 48] float32 tensor."""
    raw_img = np.random.randint(0, 256, size=(48, 48), dtype=np.uint8)
    train_t = build_train_transform()

    tensor = train_t(raw_img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 48, 48)
    assert tensor.dtype == torch.float32


def test_val_transform_determinism() -> None:
    """MANDATORY: Verify validation transform is 100% deterministic with NO random transforms."""
    raw_img = np.random.randint(0, 256, size=(48, 48), dtype=np.uint8)
    val_t = build_val_transform()

    t1 = val_t(raw_img)
    t2 = val_t(raw_img)
    t3 = val_t(raw_img)

    assert torch.equal(t1, t2), "Validation transform produced differing tensors across runs"
    assert torch.equal(t2, t3), "Validation transform produced differing tensors across runs"
    assert t1.shape == (1, 48, 48)
    assert t1.dtype == torch.float32


def test_test_transform_determinism() -> None:
    """MANDATORY: Verify test transform is 100% deterministic with NO random transforms."""
    raw_img = np.random.randint(0, 256, size=(48, 48), dtype=np.uint8)
    test_t = build_test_transform()

    t1 = test_t(raw_img)
    t2 = test_t(raw_img)

    assert torch.equal(t1, t2), "Test transform produced differing tensors across runs"
    assert t1.shape == (1, 48, 48)
    assert t1.dtype == torch.float32
