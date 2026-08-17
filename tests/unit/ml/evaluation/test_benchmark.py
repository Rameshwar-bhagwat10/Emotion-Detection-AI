"""Unit tests for inference benchmarking and parameter calculation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ml.evaluation.benchmark import (
    benchmark_model_inference,
    compute_model_statistics,
    save_benchmark_json,
    save_model_stats_json,
)


def test_compute_model_statistics() -> None:
    """Verify parameter count calculations."""
    model = nn.Sequential(
        nn.Linear(10, 5),  # 10*5 + 5 = 55
        nn.Linear(5, 2),  # 5*2 + 2 = 12 -> Total 67
    )
    # Freeze second layer
    for p in model[1].parameters():
        p.requires_grad = False

    stats = compute_model_statistics(model)
    assert stats["total_parameters"] == 67
    assert stats["trainable_parameters"] == 55
    assert stats["non_trainable_parameters"] == 12


def test_benchmark_model_inference_structure() -> None:
    """Verify benchmark output structure on CPU."""
    model = nn.Linear(4, 2)
    device = torch.device("cpu")

    results = benchmark_model_inference(
        model=model,
        device=device,
        input_shape=(4,),
        batch_size=4,
        warmup_iterations=2,
        benchmark_iterations=5,
    )

    assert "batch_benchmark" in results
    assert "single_sample_benchmark" in results
    assert results["batch_benchmark"]["mean_latency_ms"] > 0.0
    assert results["batch_benchmark"]["throughput_samples_per_sec"] > 0.0
    assert results["single_sample_benchmark"]["mean_latency_ms"] > 0.0


def test_save_benchmark_and_stats(tmp_path: Path) -> None:
    """Verify JSON persistence."""
    bench_data = {"test": 123}
    bench_path = tmp_path / "bench.json"
    save_benchmark_json(bench_data, bench_path)
    assert bench_path.exists()

    stats_data = {"total_params": 100}
    stats_path = tmp_path / "stats.json"
    save_model_stats_json(stats_data, stats_path)
    assert stats_path.exists()
