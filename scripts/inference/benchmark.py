"""Comprehensive benchmark script for Phase 09 Inference Pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.config import load_inference_config  # noqa: E402
from ml.inference.engine import EmotionInferenceEngine  # noqa: E402


def run_inference_benchmark(
    iterations: int = 50,
    warmup: int = 5,
    output_path: Path | None = None,
) -> dict:
    """Run comprehensive inference latency and throughput benchmarking."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Initializing Emotion Inference Engine for benchmarking...")

    config = load_inference_config()
    engine = EmotionInferenceEngine(config=config, auto_load=True)

    logging.info(f"Model: {engine._get_model_info().model_name} | Device: {engine.device}")

    # Generate synthetic 640x480 test image with a high-contrast facial oval to trigger Haar detection
    img_h, img_w = 480, 640
    test_img = np.full((img_h, img_w, 3), 128, dtype=np.uint8)

    # Draw simple facial feature patterns (eyes, nose, mouth oval) to allow detector testing
    import cv2

    cv2.ellipse(test_img, (320, 240), (80, 100), 0, 0, 360, (220, 200, 180), -1)
    cv2.circle(test_img, (290, 210), 12, (50, 40, 30), -1)
    cv2.circle(test_img, (350, 210), 12, (50, 40, 30), -1)
    cv2.rectangle(test_img, (315, 230), (325, 255), (70, 50, 40), -1)
    cv2.ellipse(test_img, (320, 290), (35, 15), 0, 0, 180, (40, 30, 20), -1)

    # 1. Warm-up
    logging.info(f"Running {warmup} warm-up iterations...")
    for _ in range(warmup):
        _ = engine.predict_image(test_img)

    # 2. Measurement
    logging.info(f"Executing {iterations} measurement iterations...")
    loading_times = []
    detection_times = []
    preprocessing_times = []
    inference_times = []
    postprocessing_times = []
    total_times = []
    total_faces = 0

    t_start_suite = time.perf_counter()
    for _ in range(iterations):
        res = engine.predict_image(test_img)
        if res.timing is not None:
            loading_times.append(res.timing.image_loading_ms)
            detection_times.append(res.timing.face_detection_ms)
            preprocessing_times.append(res.timing.preprocessing_ms)
            inference_times.append(res.timing.inference_ms)
            postprocessing_times.append(res.timing.postprocessing_ms)
            total_times.append(res.timing.total_ms)
        total_faces += res.faces_detected

    total_elapsed_sec = time.perf_counter() - t_start_suite

    # Calculate statistics
    avg_load = float(np.mean(loading_times)) if loading_times else 0.0
    avg_det = float(np.mean(detection_times)) if detection_times else 0.0
    avg_prep = float(np.mean(preprocessing_times)) if preprocessing_times else 0.0
    avg_inf = float(np.mean(inference_times)) if inference_times else 0.0
    avg_post = float(np.mean(postprocessing_times)) if postprocessing_times else 0.0
    avg_total = float(np.mean(total_times)) if total_times else 0.0
    median_total = float(np.median(total_times)) if total_times else 0.0
    p95_total = float(np.percentile(total_times, 95)) if total_times else 0.0

    throughput_fps = round(iterations / total_elapsed_sec, 2)
    faces_per_sec = round(total_faces / total_elapsed_sec, 2) if total_faces > 0 else throughput_fps

    results = {
        "model_name": engine._get_model_info().model_name,
        "architecture": engine._get_model_info().architecture,
        "optimization_type": engine._get_model_info().optimization_type,
        "device": str(engine.device),
        "detector": config.face_detection.detector_type,
        "image_size": f"{img_w}x{img_h}",
        "iterations": iterations,
        "warmup": warmup,
        "latency_ms": {
            "image_loading_avg": round(avg_load, 2),
            "face_detection_avg": round(avg_det, 2),
            "preprocessing_avg": round(avg_prep, 2),
            "inference_avg": round(avg_inf, 2),
            "postprocessing_avg": round(avg_post, 2),
            "total_mean": round(avg_total, 2),
            "total_median": round(median_total, 2),
            "total_p95": round(p95_total, 2),
        },
        "throughput": {
            "images_per_second": throughput_fps,
            "faces_per_second": faces_per_sec,
        },
    }

    print("\n" + "=" * 80)
    print("PHASE 09 INFERENCE ENGINE BENCHMARK REPORT")
    print("=" * 80)
    print(
        f"Model:                {results['model_name']} ({results['architecture']}, {results['optimization_type']})"
    )
    print(f"Device:               {results['device']}")
    print(f"Face Detector:        {results['detector']}")
    print(f"Image Resolution:     {results['image_size']}")
    print(f"Measurement Runs:     {iterations} iterations")
    print("-" * 80)
    print(f"Image Loading:        {avg_load:.2f} ms")
    print(f"Face Detection:       {avg_det:.2f} ms")
    print(f"Preprocessing:        {avg_prep:.2f} ms")
    print(f"Model Inference:      {avg_inf:.2f} ms")
    print(f"Postprocessing:       {avg_post:.2f} ms")
    print("-" * 80)
    print(f"Total Pipeline Mean:  {avg_total:.2f} ms")
    print(f"Total Median Latency: {median_total:.2f} ms")
    print(f"Total P95 Latency:    {p95_total:.2f} ms")
    print(f"Throughput:           {throughput_fps} images/sec ({faces_per_sec} faces/sec)")
    print("=" * 80 + "\n")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved benchmark report to {out_p}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 09 inference benchmark")
    parser.add_argument("--iterations", type=int, default=30, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/benchmark/inference_benchmark.json",
        help="Path to save benchmark JSON",
    )
    args = parser.parse_args()

    out_file = (
        PROJECT_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    )
    run_inference_benchmark(iterations=args.iterations, warmup=args.warmup, output_path=out_file)
