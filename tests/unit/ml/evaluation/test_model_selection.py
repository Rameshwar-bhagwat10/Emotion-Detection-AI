"""Unit tests for model candidate comparison and champion selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.evaluation.selection import (
    DecisionWeights,
    ModelCandidate,
    save_model_selection_artifacts,
    select_champion_model,
)


def test_compare_candidates_ranking() -> None:
    """Verify multi-criteria candidate comparison ranks strongest candidate first."""
    candidates = [
        ModelCandidate(
            model_name="baseline_cnn",
            checkpoint_path="baseline/best.pt",
            val_accuracy=0.2951,
            val_macro_f1=0.1700,
            val_weighted_f1=0.2100,
            val_loss=1.7131,
            batch_latency_ms=26.0,
            single_sample_latency_ms=0.85,
            throughput_fps=2400.0,
            total_parameters=422119,
            model_size_mb=4.85,
        ),
        ModelCandidate(
            model_name="resnet18",
            checkpoint_path="resnet18/best.pt",
            val_accuracy=0.4500,
            val_macro_f1=0.4100,
            val_weighted_f1=0.4300,
            val_loss=1.4200,
            batch_latency_ms=35.0,
            single_sample_latency_ms=1.50,
            throughput_fps=1800.0,
            total_parameters=11176519,
            model_size_mb=42.7,
        ),
        ModelCandidate(
            model_name="mobilenet_v3_small",
            checkpoint_path="mobilenet/best.pt",
            val_accuracy=0.3800,
            val_macro_f1=0.3200,
            val_weighted_f1=0.3500,
            val_loss=1.5500,
            batch_latency_ms=18.0,
            single_sample_latency_ms=0.90,
            throughput_fps=3200.0,
            total_parameters=1522855,
            model_size_mb=6.2,
        ),
    ]

    champion, ranked_df = select_champion_model(candidates)

    assert champion.model_name == "resnet18"
    assert ranked_df.iloc[0]["rank"] == 1
    assert ranked_df.iloc[0]["model_name"] == "resnet18"
    assert len(ranked_df) == 3


def test_save_model_selection_artifacts(tmp_path: Path) -> None:
    """Verify selection artifacts are serialized to CSV, JSON, and Markdown."""
    candidate = ModelCandidate(
        model_name="resnet18",
        checkpoint_path="resnet18/best.pt",
        val_accuracy=0.50,
        val_macro_f1=0.48,
        val_weighted_f1=0.49,
        val_loss=1.35,
        batch_latency_ms=30.0,
        single_sample_latency_ms=1.2,
        throughput_fps=2000.0,
        total_parameters=11000000,
        model_size_mb=42.0,
    )

    champion, ranked_df = select_champion_model([candidate])
    save_model_selection_artifacts(ranked_df, champion, tmp_path)

    assert (tmp_path / "model_selection_matrix.csv").exists()
    assert (tmp_path / "model_selection_matrix.json").exists()
    assert (tmp_path / "model_selection_summary.md").exists()


def test_decision_weights_validation() -> None:
    """Verify validation error when weights do not sum to 1.0."""
    with pytest.raises(ValueError, match="Decision weights must sum to 1.0"):
        DecisionWeights(accuracy_weight=0.5, macro_f1_weight=0.6)
