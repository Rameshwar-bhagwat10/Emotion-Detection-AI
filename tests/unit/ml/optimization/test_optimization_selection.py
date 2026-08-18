"""Unit tests for Quality Gate and Optimized Champion selection."""

from __future__ import annotations

from pathlib import Path

from ml.optimization.config import QualityGateConfig
from ml.optimization.selection import (
    OptimizationCandidate,
    evaluate_quality_gate,
    save_optimization_artifacts,
    select_optimized_champion,
)


def test_evaluate_quality_gate_acceptance_and_rejection() -> None:
    """Verify Quality Gate accepts candidates within thresholds and rejects degraded candidates."""
    ref = OptimizationCandidate(
        candidate_id="champion-fp32",
        base_model="resnet18",
        optimization_type="fp32_reference",
        val_accuracy=0.5200,
        val_macro_f1=0.4300,
        val_weighted_f1=0.5000,
        latency_ms=3.10,
        throughput_fps=1400.0,
        model_size_mb=128.0,
        total_parameters=11180103,
    )

    # Candidate A: good F1, smaller size -> Accepted
    cand_a = OptimizationCandidate(
        candidate_id="champion-int8",
        base_model="resnet18",
        optimization_type="int8_ptq",
        val_accuracy=0.5150,
        val_macro_f1=0.4250,
        val_weighted_f1=0.4950,
        latency_ms=2.80,
        throughput_fps=1500.0,
        model_size_mb=32.0,  # 75% size reduction
        total_parameters=11180103,
    )

    # Candidate B: severe degradation -> Rejected
    cand_b = OptimizationCandidate(
        candidate_id="champion-bad",
        base_model="resnet18",
        optimization_type="aggressive_pruning",
        val_accuracy=0.4000,
        val_macro_f1=0.3000,
        val_weighted_f1=0.3800,
        latency_ms=2.00,
        throughput_fps=1800.0,
        model_size_mb=20.0,
        total_parameters=11180103,
    )

    cfg = QualityGateConfig(max_macro_f1_drop=0.03, max_accuracy_drop=0.03)

    is_acc_a, _ = evaluate_quality_gate(ref, cand_a, cfg)
    is_acc_b, _ = evaluate_quality_gate(ref, cand_b, cfg)

    assert is_acc_a is True
    assert is_acc_b is False


def test_select_optimized_champion_fallback_to_reference(tmp_path: Path) -> None:
    """Verify fallback to FP32 reference when all candidates fail quality gate."""
    ref = OptimizationCandidate(
        candidate_id="champion-fp32",
        base_model="resnet18",
        optimization_type="fp32_reference",
        val_accuracy=0.5200,
        val_macro_f1=0.4300,
        val_weighted_f1=0.5000,
        latency_ms=3.10,
        throughput_fps=1400.0,
        model_size_mb=128.0,
        total_parameters=11180103,
    )

    bad_cand = OptimizationCandidate(
        candidate_id="bad-cand",
        base_model="resnet18",
        optimization_type="pruning_90",
        val_accuracy=0.3000,
        val_macro_f1=0.2000,
        val_weighted_f1=0.2500,
        latency_ms=1.00,
        throughput_fps=2000.0,
        model_size_mb=10.0,
        total_parameters=11180103,
    )

    champion, df = select_optimized_champion(ref, [bad_cand])
    assert champion.candidate_id == "champion-fp32"

    save_optimization_artifacts(df, champion, tmp_path)
    assert (tmp_path / "optimization_matrix.csv").exists()
    assert (tmp_path / "optimization_matrix.json").exists()
    assert (tmp_path / "selection_report.md").exists()
