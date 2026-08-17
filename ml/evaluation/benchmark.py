"""Model inference latency, throughput benchmarking, and parameter statistics."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def compute_model_statistics(
    model: nn.Module,
    checkpoint_path: Path | str | None = None,
) -> dict[str, Any]:
    """Calculate parameter counts and filesystem checkpoint size.

    Args:
        model: PyTorch model instance.
        checkpoint_path: Optional path to saved .pt checkpoint file.

    Returns:
        Dictionary with total, trainable, non-trainable parameter counts and file sizes.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    ckpt_size_bytes = 0
    ckpt_size_mb = 0.0

    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if p.exists():
            ckpt_size_bytes = p.stat().st_size
            ckpt_size_mb = round(ckpt_size_bytes / (1024 * 1024), 4)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "checkpoint_size_bytes": ckpt_size_bytes,
        "checkpoint_size_mb": ckpt_size_mb,
    }


def benchmark_model_inference(
    model: nn.Module,
    device: torch.device,
    input_shape: tuple[int, int, int] = (1, 48, 48),
    batch_size: int = 64,
    warmup_iterations: int = 10,
    benchmark_iterations: int = 50,
) -> dict[str, Any]:
    """Benchmark inference latency (batch and single-sample) and throughput.

    Args:
        model: PyTorch model in evaluation mode.
        device: Device to benchmark on (CPU or CUDA).
        input_shape: (C, H, W) shape of input images.
        batch_size: Evaluation batch size.
        warmup_iterations: Number of warmup inferences before measurement.
        benchmark_iterations: Number of timed iterations.

    Returns:
        Structured dictionary with latency and throughput metrics.
    """
    model.eval()
    model.to(device)

    is_cuda = device.type == "cuda"

    # 1. Batch inference benchmark
    batch_dummy = torch.randn(batch_size, *input_shape, device=device, dtype=torch.float32)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_iterations):
            _ = model(batch_dummy)
            if is_cuda:
                torch.cuda.synchronize()

        # Timed iterations
        batch_latencies_ms: list[float] = []
        for _ in range(benchmark_iterations):
            t_start = time.perf_counter()
            _ = model(batch_dummy)
            if is_cuda:
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            batch_latencies_ms.append((t_end - t_start) * 1000.0)

    avg_batch_latency_ms = float(np.mean(batch_latencies_ms))
    median_batch_latency_ms = float(np.median(batch_latencies_ms))
    p95_batch_latency_ms = float(np.percentile(batch_latencies_ms, 95))
    throughput_samples_per_sec = (
        float((batch_size / avg_batch_latency_ms) * 1000.0) if avg_batch_latency_ms > 0 else 0.0
    )

    # 2. Single-sample (batch_size=1) real-time latency benchmark
    single_dummy = torch.randn(1, *input_shape, device=device, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(single_dummy)
            if is_cuda:
                torch.cuda.synchronize()

        single_latencies_ms: list[float] = []
        for _ in range(benchmark_iterations):
            t_start = time.perf_counter()
            _ = model(single_dummy)
            if is_cuda:
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            single_latencies_ms.append((t_end - t_start) * 1000.0)

    avg_single_latency_ms = float(np.mean(single_latencies_ms))
    median_single_latency_ms = float(np.median(single_latencies_ms))
    p95_single_latency_ms = float(np.percentile(single_latencies_ms, 95))

    return {
        "device": device.type,
        "warmup_iterations": warmup_iterations,
        "benchmark_iterations": benchmark_iterations,
        "batch_benchmark": {
            "batch_size": batch_size,
            "mean_latency_ms": round(avg_batch_latency_ms, 4),
            "median_latency_ms": round(median_batch_latency_ms, 4),
            "p95_latency_ms": round(p95_batch_latency_ms, 4),
            "throughput_samples_per_sec": round(throughput_samples_per_sec, 2),
        },
        "single_sample_benchmark": {
            "batch_size": 1,
            "mean_latency_ms": round(avg_single_latency_ms, 4),
            "median_latency_ms": round(median_single_latency_ms, 4),
            "p95_latency_ms": round(p95_single_latency_ms, 4),
            "throughput_samples_per_sec": (
                round((1000.0 / avg_single_latency_ms), 2) if avg_single_latency_ms > 0 else 0.0
            ),
        },
    }


def save_benchmark_json(
    benchmark_data: dict[str, Any],
    output_path: Path,
) -> None:
    """Serialize benchmark metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)


def save_model_stats_json(
    stats_data: dict[str, Any],
    output_path: Path,
) -> None:
    """Serialize model parameter and size stats to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)
