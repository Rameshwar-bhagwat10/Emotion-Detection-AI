"""Unit tests for early stopping logic and state preservation."""

from __future__ import annotations

import pytest

from ml.training.early_stopping import EarlyStopping


def test_early_stopping_min_mode_trigger() -> None:
    """Verify early stopping triggers after patience non-improving epochs in min mode."""
    es = EarlyStopping(patience=3, min_delta=0.01, mode="min")

    # Epoch 1: 1.0 (new best)
    is_imp, should_stop = es.step(1.0, epoch=1)
    assert is_imp is True and should_stop is False
    assert es.patience_counter == 0
    assert es.best_score == 1.0

    # Epoch 2: 0.995 (improvement < min_delta of 0.01 -> non-improvement)
    is_imp, should_stop = es.step(0.995, epoch=2)
    assert is_imp is False and should_stop is False
    assert es.patience_counter == 1

    # Epoch 3: 1.05 (worse)
    is_imp, should_stop = es.step(1.05, epoch=3)
    assert is_imp is False and should_stop is False
    assert es.patience_counter == 2

    # Epoch 4: 1.02 (worse, hits patience 3 -> triggers stop)
    is_imp, should_stop = es.step(1.02, epoch=4)
    assert is_imp is False and should_stop is True
    assert es.is_triggered is True


def test_early_stopping_reset_on_improvement() -> None:
    """Verify patience counter resets to 0 when significant improvement occurs."""
    es = EarlyStopping(patience=3, min_delta=0.01, mode="min")

    es.step(1.0, epoch=1)
    es.step(1.1, epoch=2)  # counter = 1
    es.step(1.2, epoch=3)  # counter = 2
    assert es.patience_counter == 2

    # Significant improvement
    is_imp, should_stop = es.step(0.8, epoch=4)
    assert is_imp is True and should_stop is False
    assert es.patience_counter == 0
    assert es.best_score == 0.8


def test_early_stopping_max_mode() -> None:
    """Verify early stopping in max mode (e.g. accuracy)."""
    es = EarlyStopping(patience=2, min_delta=0.05, mode="max")

    es.step(0.5, epoch=1)
    assert es.best_score == 0.5

    # 0.56 (improvement > 0.05)
    is_imp, should_stop = es.step(0.56, epoch=2)
    assert is_imp is True and should_stop is False

    # 0.57 (improvement < 0.05) -> patience 1
    is_imp, should_stop = es.step(0.57, epoch=3)
    assert is_imp is False and should_stop is False

    # 0.50 (worse) -> patience 2 -> triggers stop
    is_imp, should_stop = es.step(0.50, epoch=4)
    assert is_imp is False and should_stop is True


def test_early_stopping_disabled() -> None:
    """Verify early stopping never triggers when enabled is False."""
    es = EarlyStopping(patience=1, enabled=False)
    for i in range(10):
        is_imp, should_stop = es.step(100.0, epoch=i)
        assert should_stop is False


def test_early_stopping_validation() -> None:
    """Verify parameter validation."""
    with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
        EarlyStopping(mode="invalid")
    with pytest.raises(ValueError, match="patience must be >= 0"):
        EarlyStopping(patience=-1)
    with pytest.raises(ValueError, match="min_delta must be >= 0.0"):
        EarlyStopping(min_delta=-0.1)
