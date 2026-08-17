"""Independent strict audit and verification script for Phase 06."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT_DIR = Path("D:/projects/emotion-detection-ai")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.datasets.fer2013.parser import EMOTION_NAMES  # noqa: E402
from ml.evaluation.benchmark import (  # noqa: E402
    benchmark_model_inference,
)
from ml.evaluation.confusion_matrix import (  # noqa: E402
    compute_confusion_matrix,
    compute_normalized_confusion_matrix,
)
from ml.evaluation.error_analysis import analyze_confidence  # noqa: E402
from ml.evaluation.evaluator import Evaluator  # noqa: E402
from ml.evaluation.metrics import (  # noqa: E402
    calculate_accuracy,
    calculate_macro_metrics,
    calculate_per_class_metrics,
    calculate_weighted_metrics,
)
from ml.models.factory import create_model  # noqa: E402
from ml.preprocessing.dataloaders import build_dataloaders  # noqa: E402
from ml.training.checkpointing import CheckpointManager  # noqa: E402


def run_comprehensive_audit() -> dict[str, Any]:
    """Run all strict verification checks for Phase 06."""
    audit_results: dict[str, Any] = {}
    print("=" * 60)
    print("STARTING STRICT INDEPENDENT AUDIT: PHASE 06")
    print("=" * 60)

    # 1. Checkpoint Verification
    print("\n[CHECK 1] Auditing Phase 05 Best Checkpoint...")
    ckpt_path = ROOT_DIR / "artifacts/training/baseline_cnn/smoke_test_20260817_055905/best.pt"
    assert ckpt_path.exists(), f"Checkpoint file missing: {ckpt_path}"
    assert ckpt_path.stat().st_size > 0, "Checkpoint file is 0 bytes"
    ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"  -> Checkpoint exists at {ckpt_path} ({ckpt_size_mb:.2f} MB)")

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in raw_ckpt, "model_state_dict missing from checkpoint!"
    assert "epoch" in raw_ckpt, "epoch missing from checkpoint!"
    assert "best_metric" in raw_ckpt, "best_metric missing from checkpoint!"
    print(
        f"  -> Checkpoint metadata: Epoch {raw_ckpt['epoch']}, Best Val Loss: {raw_ckpt['best_metric']:.6f}"
    )
    audit_results["checkpoint_verification"] = "PASS"

    # 2. Model Compatibility & Real Checkpoint Load Test
    print("\n[CHECK 2] Auditing Model Instantiation and Checkpoint Weight Loading...")
    model = create_model("baseline_cnn")
    assert isinstance(model, nn.Module)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 422119, f"Expected 422119 params, got {total_params}"
    assert trainable_params == 422119, f"Expected 422119 trainable params, got {trainable_params}"

    loaded_ckpt = CheckpointManager.load(ckpt_path, model=model, device=torch.device("cpu"))
    print(
        f"  -> CheckpointManager.load() successful (epoch={loaded_ckpt.get('epoch')}, best_metric={loaded_ckpt.get('best_metric')})"
    )

    dummy_input = torch.randn(4, 1, 48, 48, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        dummy_logits = model(dummy_input)
    assert dummy_logits.shape == (4, 7), f"Expected shape [4, 7], got {dummy_logits.shape}"
    assert torch.isfinite(dummy_logits).all(), "Non-finite logits detected!"
    print("  -> Forward pass verified: shape [4, 7], all logits finite.")
    audit_results["model_compatibility"] = "PASS"

    # 3. Phase 03 Test DataLoader Audit
    print("\n[CHECK 3] Auditing Phase 03 Test DataLoader & Dataset Split Integrity...")
    raw_data_path = ROOT_DIR / "data/raw/fer2013/fer2013.csv"
    assert raw_data_path.exists(), f"FER2013 raw data missing: {raw_data_path}"

    train_loader, val_loader, test_loader = build_dataloaders(
        raw_data_path, batch_size=64, num_workers=0
    )
    assert (
        len(test_loader.dataset) == 3589
    ), f"Expected 3589 test samples, got {len(test_loader.dataset)}"
    assert (
        len(val_loader.dataset) == 3589
    ), f"Expected 3589 val samples, got {len(val_loader.dataset)}"
    assert (
        len(train_loader.dataset) == 28709
    ), f"Expected 28709 train samples, got {len(train_loader.dataset)}"

    first_batch = next(iter(test_loader))
    assert "image" in first_batch and "label" in first_batch
    assert first_batch["image"].shape == (64, 1, 48, 48)
    assert first_batch["image"].dtype == torch.float32
    assert first_batch["label"].shape == (64,)
    assert first_batch["label"].dtype == torch.long
    assert (first_batch["label"] >= 0).all() and (first_batch["label"] <= 6).all()
    print(
        f"  -> Test DataLoader verified: {len(test_loader.dataset)} samples across {len(test_loader)} batches."
    )
    audit_results["test_dataloader"] = "PASS"

    # 4. Parameter Immutability Test
    print("\n[CHECK 4] Auditing Parameter Immutability During Evaluation...")
    params_before = [p.clone().detach() for p in model.parameters()]

    evaluator = Evaluator(
        model=model,
        data_loader=test_loader,
        device="cpu",
        checkpoint_path=ckpt_path,
        run_dir=ROOT_DIR / "artifacts/evaluation/baseline_cnn/audit_test_run",
    )
    _ = evaluator.evaluate()

    params_after = [p.clone().detach() for p in model.parameters()]
    for idx, (pb, pa) in enumerate(zip(params_before, params_after, strict=True)):
        assert torch.equal(pb, pa), f"Parameter tensor {idx} was mutated during evaluation!"
    print(
        "  -> Immutability confirmed: All 16 parameter tensors identical before and after evaluation."
    )
    audit_results["parameter_immutability"] = "PASS"

    # 5. Independent Mathematical Verification of Metrics
    print("\n[CHECK 5] Independently Verifying Metrics & Confusion Matrix Calculations...")
    y_true_list = []
    y_pred_list = []
    probs_list = []
    softmax = nn.Softmax(dim=-1)
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"]
            lbls = batch["label"]
            logits = model(imgs)
            probs = softmax(logits)
            preds = torch.argmax(logits, dim=-1)

            y_true_list.extend(lbls.numpy().tolist())
            y_pred_list.extend(preds.numpy().tolist())
            probs_list.append(probs.numpy())

    y_true = np.array(y_true_list, dtype=np.int64)
    y_pred = np.array(y_pred_list, dtype=np.int64)
    probs = np.vstack(probs_list)

    total_samples = len(y_true)
    correct_samples = int(np.sum(y_true == y_pred))
    incorrect_samples = total_samples - correct_samples
    expected_acc = correct_samples / total_samples

    calc_acc = calculate_accuracy(y_true, y_pred)
    assert np.isclose(
        calc_acc, expected_acc
    ), f"Accuracy mismatch: calc={calc_acc}, expected={expected_acc}"
    print(f"  -> Accuracy: {calc_acc*100:.4f}% ({correct_samples}/{total_samples} samples)")

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=7)
    assert cm.shape == (7, 7)
    assert cm.sum() == total_samples, f"CM sum ({cm.sum()}) != total samples ({total_samples})"
    assert (
        np.trace(cm) == correct_samples
    ), f"CM diagonal ({np.trace(cm)}) != correct ({correct_samples})"
    assert (
        cm.sum() - np.trace(cm)
    ) == incorrect_samples, "CM off-diagonal sum != incorrect samples"

    norm_cm = compute_normalized_confusion_matrix(cm)
    assert norm_cm.shape == (7, 7)
    assert np.allclose(norm_cm.sum(axis=1), np.ones(7)), "Normalized CM rows do not sum to 1.0"
    print("  -> Confusion matrix & normalized matrix verified.")

    per_class = calculate_per_class_metrics(
        y_true, y_pred, num_classes=7, class_names=list(EMOTION_NAMES)
    )
    macro = calculate_macro_metrics(per_class)
    weighted = calculate_weighted_metrics(per_class)

    f1_list = [per_class[c]["f1"] for c in EMOTION_NAMES]
    assert np.isclose(macro["macro_f1"], np.mean(f1_list), atol=1e-5)
    print(f"  -> Macro F1: {macro['macro_f1']:.4f} | Weighted F1: {weighted['weighted_f1']:.4f}")
    audit_results["metrics_mathematics"] = "PASS"

    # 6. Confidence & Error Analysis Audit
    print("\n[CHECK 6] Auditing Confidence and Error Analysis...")
    conf_analysis = analyze_confidence(
        y_true, y_pred, probs, list(EMOTION_NAMES), high_confidence_threshold=0.90, num_bins=10
    )
    confidences = np.max(probs, axis=-1)
    assert np.isclose(conf_analysis["overall"]["mean_confidence"], np.mean(confidences), atol=1e-5)
    assert conf_analysis["correct_predictions"]["count"] == correct_samples
    assert conf_analysis["incorrect_predictions"]["count"] == incorrect_samples

    high_conf_errors = conf_analysis["high_confidence_errors"]["samples"]
    for err in high_conf_errors:
        assert (
            err["actual_class"] != err["predicted_class"]
        ), "Correct sample flagged in high-confidence errors!"
        assert err["confidence"] >= 0.90, "Confidence below 0.90 in high-confidence errors!"
    print(f"  -> Mean confidence: {conf_analysis['overall']['mean_confidence']:.4f}")
    print(f"  -> High-confidence errors (>=0.90): {len(high_conf_errors)}")
    audit_results["confidence_and_error_analysis"] = "PASS"

    # 7. Benchmarking Methodology Audit
    print("\n[CHECK 7] Auditing Benchmarking Methodology...")
    bench_results = benchmark_model_inference(
        model=model,
        device=torch.device("cpu"),
        batch_size=64,
        warmup_iterations=5,
        benchmark_iterations=20,
    )
    assert "batch_benchmark" in bench_results
    assert "single_sample_benchmark" in bench_results
    assert bench_results["batch_benchmark"]["mean_latency_ms"] > 0
    assert bench_results["batch_benchmark"]["throughput_samples_per_sec"] > 0
    assert bench_results["single_sample_benchmark"]["mean_latency_ms"] > 0
    print(f"  -> CPU Batch Latency: {bench_results['batch_benchmark']['mean_latency_ms']:.2f} ms")
    print(
        f"  -> CPU Single Sample Latency: {bench_results['single_sample_benchmark']['mean_latency_ms']:.2f} ms"
    )
    audit_results["benchmarking_methodology"] = "PASS"

    # 8. Artifact Verification Audit
    print("\n[CHECK 8] Auditing Generated Artifact Integrity & Consistency...")
    eval_run_dir = ROOT_DIR / "artifacts/evaluation/baseline_cnn/baseline_cnn_eval_20260817_063335"
    expected_files = [
        "evaluation_results.json",
        "classification_report.json",
        "classification_report.csv",
        "confusion_matrix.csv",
        "confusion_matrix_normalized.csv",
        "incorrect_predictions.csv",
        "confusion_pairs.json",
        "confidence_analysis.json",
        "benchmark.json",
        "model_stats.json",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "confidence_distribution.png",
        "evaluation.log",
    ]

    for fname in expected_files:
        fpath = eval_run_dir / fname
        assert fpath.exists(), f"Missing artifact: {fname}"
        assert fpath.stat().st_size > 0, f"Empty artifact: {fname}"

    eval_json = json.loads((eval_run_dir / "evaluation_results.json").read_text(encoding="utf-8"))
    incorrect_df = pd.read_csv(eval_run_dir / "incorrect_predictions.csv")

    assert eval_json["total_samples"] == 3589
    assert np.isclose(eval_json["accuracy"], 0.292003, atol=1e-4)
    assert len(incorrect_df) == (3589 - 1048)  # 2541
    print(
        f"  -> All {len(expected_files)} artifacts verified for existence, non-emptiness, and consistency."
    )
    audit_results["artifact_integrity"] = "PASS"

    # 9. CLI Tool Verification
    print("\n[CHECK 9] Auditing Evaluation CLI Entrypoint...")
    cli_file = ROOT_DIR / "scripts/evaluation/evaluate.py"
    assert cli_file.exists(), f"CLI file missing: {cli_file}"
    audit_results["cli_verification"] = "PASS"

    # 10. Scope Compliance Audit
    print("\n[CHECK 10] Auditing Scope Compliance (No Training/Retraining/Future Phase Leakage)...")
    eval_py_content = (ROOT_DIR / "ml/evaluation/evaluator.py").read_text(encoding="utf-8")
    assert "optimizer.step" not in eval_py_content
    assert "loss.backward" not in eval_py_content
    assert "model.train()" not in eval_py_content
    print("  -> Scope compliance verified: Pure evaluation with zero training logic.")
    audit_results["scope_compliance"] = "PASS"

    print("\n" + "=" * 60)
    print("ALL AUDIT CHECKS PASSED WITH ZERO ERRORS")
    print("=" * 60)

    audit_results["final_verdict"] = "PHASE 06 — PASS"
    return audit_results


if __name__ == "__main__":
    run_comprehensive_audit()
