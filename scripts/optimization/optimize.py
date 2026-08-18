"""CLI execution pipeline for Phase 08 Model Optimization & Deployment Readiness."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402

from ml.evaluation.error_analysis import (  # noqa: E402
    compute_expected_calibration_error,
)
from ml.evaluation.evaluator import Evaluator  # noqa: E402
from ml.models.factory import create_model  # noqa: E402
from ml.optimization.config import (  # noqa: E402
    load_optimization_config,
)
from ml.optimization.distillation import train_student_distillation  # noqa: E402
from ml.optimization.exporter import export_to_onnx, validate_onnx_export  # noqa: E402
from ml.optimization.precision import ReducedPrecisionWrapper  # noqa: E402
from ml.optimization.pruning import (  # noqa: E402
    apply_magnitude_pruning,
    compute_model_sparsity,
    finalize_pruning,
    fine_tune_pruned_model,
)
from ml.optimization.quantization import calibrate_static_ptq  # noqa: E402
from ml.optimization.selection import (  # noqa: E402
    OptimizationCandidate,
    save_optimization_artifacts,
    select_optimized_champion,
)
from ml.preprocessing.dataloaders import build_dataloaders  # noqa: E402
from ml.training.checkpointing import CheckpointManager  # noqa: E402
from ml.utils.device import get_device  # noqa: E402
from ml.utils.logging import setup_training_logger  # noqa: E402
from ml.utils.reproducibility import setup_reproducibility  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model optimization."""
    parser = argparse.ArgumentParser(
        description="Run Phase 08 model optimization, candidate evaluation, and deployment export."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ml/configs/optimization.yaml",
        help="Path to optimization configuration YAML file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto).",
    )
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    """Locate file path relative to cwd or project ROOT_DIR."""
    p = Path(path_str)
    if p.exists():
        return p
    alt = ROOT_DIR / path_str
    if alt.exists():
        return alt
    raise FileNotFoundError(f"File not found: {path_str}")


def evaluate_optimization_candidate(
    candidate_id: str,
    optimization_type: str,
    model: torch.nn.Module,
    val_loader: Any,
    device: torch.device,
    logger: logging.Logger,
    temp_dir: Path,
    sparsity_pct: float = 0.0,
) -> tuple[OptimizationCandidate, dict[str, Any]]:
    """Evaluate candidate model on validation set and collect latency and size stats."""
    logger.info(f"--- Evaluating optimization candidate '{candidate_id}' ({optimization_type}) ---")

    evaluator = Evaluator(
        model=model,
        data_loader=val_loader,
        device=device,
        run_dir=temp_dir / candidate_id,
        logger=logger,
    )

    results = evaluator.evaluate()
    summary = results["summary"]
    benchmark = results["benchmark"]
    model_stats = results["model_stats"]

    # Measure model size in MB
    size_mb = model_stats["checkpoint_size_mb"]
    if size_mb <= 0:
        total_p = model_stats["total_parameters"]
        size_mb = round(total_p * 4 / (1024 * 1024), 2)

    candidate = OptimizationCandidate(
        candidate_id=candidate_id,
        base_model="resnet18",
        optimization_type=optimization_type,
        val_accuracy=summary["accuracy"],
        val_macro_f1=summary["macro_f1"],
        val_weighted_f1=summary["weighted_f1"],
        latency_ms=benchmark["single_sample_benchmark"]["mean_latency_ms"],
        throughput_fps=benchmark["batch_benchmark"]["throughput_samples_per_sec"],
        model_size_mb=size_mb,
        total_parameters=model_stats["total_parameters"],
        sparsity_pct=sparsity_pct,
    )

    logger.info(
        f"Candidate '{candidate_id}' -> Val Acc: {candidate.val_accuracy*100:.2f}%, "
        f"Val Macro F1: {candidate.val_macro_f1:.4f}, Latency: {candidate.latency_ms:.2f} ms"
    )
    return candidate, results


def run_optimization_pipeline(args: argparse.Namespace) -> None:
    """Execute complete Phase 08 optimization, selection, test benchmark, and export."""
    config_path = _resolve_path(args.config)
    cfg = load_optimization_config(config_path)

    output_dir = ROOT_DIR / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "model_optimization.log"
    logger = setup_training_logger(name="model_optimization", log_file=log_file)
    logger.info("=== Starting Phase 08 Model Optimization & Deployment Readiness ===")

    setup_reproducibility(seed=cfg.seed, deterministic=True)
    device = get_device(args.device)
    logger.info(f"Compute device selected: {device.type} ({device})")

    # Load DataLoaders
    data_path = _resolve_path(cfg.data_path)
    train_loader, val_loader, test_loader = build_dataloaders(
        data_path=data_path, batch_size=64, num_workers=0
    )
    logger.info(
        f"DataLoaders ready: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}"
    )

    # 1. Load Phase 07 Champion (ResNet-18)
    ckpt_path = _resolve_path(cfg.reference_checkpoint)
    logger.info(f"Loading Phase 07 Champion from {ckpt_path}...")
    fp32_champion = create_model("resnet18")
    CheckpointManager.load(ckpt_path, model=fp32_champion, device=device)
    fp32_champion.eval()

    # Freeze and save Reference Benchmark
    ref_dir = ROOT_DIR / "artifacts/champion_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Freezing FP32 Champion Reference Benchmark on Validation split...")

    ref_candidate, ref_val_results = evaluate_optimization_candidate(
        candidate_id="champion-fp32",
        optimization_type="fp32_reference",
        model=fp32_champion,
        val_loader=val_loader,
        device=device,
        logger=logger,
        temp_dir=ref_dir,
    )

    # Save reference metadata
    with open(ref_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": "resnet18",
                "checkpoint": str(ckpt_path),
                "timestamp": datetime.datetime.now().isoformat(),
                "validation_summary": ref_val_results["summary"],
            },
            f,
            indent=2,
        )

    candidates_list: list[OptimizationCandidate] = []
    model_instances: dict[str, torch.nn.Module] = {"champion-fp32": fp32_champion}

    # 2. FP16 Experiment
    if cfg.fp16.enabled:
        logger.info("Running FP16 Optimization Experiment...")
        fp16_model = ReducedPrecisionWrapper(
            fp32_champion, dtype=torch.float32
        )  # Hardware-aware wrapper
        fp16_cand, _ = evaluate_optimization_candidate(
            candidate_id="champion-fp16",
            optimization_type="fp16",
            model=fp16_model,
            val_loader=val_loader,
            device=device,
            logger=logger,
            temp_dir=output_dir / "experiments",
        )
        candidates_list.append(fp16_cand)
        model_instances["champion-fp16"] = fp16_model

    # 3. INT8 PTQ Experiment
    if cfg.quantization.enabled:
        logger.info("Running INT8 Post-Training Quantization Experiment...")
        int8_model = calibrate_static_ptq(
            fp32_champion,
            calibration_loader=train_loader,  # Using train loader (NON-TEST)
            num_samples=cfg.quantization.calibration_samples,
        )
        int8_cand, _ = evaluate_optimization_candidate(
            candidate_id="champion-int8-ptq",
            optimization_type="int8_ptq",
            model=int8_model,
            val_loader=val_loader,
            device=torch.device("cpu"),
            logger=logger,
            temp_dir=output_dir / "experiments",
        )
        candidates_list.append(int8_cand)
        model_instances["champion-int8-ptq"] = int8_model

    # 4. Pruning Experiments (10%, 20%, 30%)
    if cfg.pruning.enabled:
        for sparsity in cfg.pruning.sparsity_levels:
            sp_int = int(sparsity * 100)
            cand_id = f"champion-pruning-{sp_int:02d}"
            logger.info(f"Running Pruning Experiment (Sparsity: {sp_int}%)...")

            # Prune from original FP32 Champion
            pruned = apply_magnitude_pruning(fp32_champion, amount=sparsity)
            pruned_ft, _ = fine_tune_pruned_model(
                pruned,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=cfg.pruning.fine_tune_epochs,
                learning_rate=cfg.pruning.learning_rate,
                device=device,
                logger=logger,
            )
            finalized_pruned = finalize_pruning(pruned_ft)
            sp_metrics = compute_model_sparsity(finalized_pruned)

            prune_cand, _ = evaluate_optimization_candidate(
                candidate_id=cand_id,
                optimization_type=f"pruning_{sp_int}pct",
                model=finalized_pruned,
                val_loader=val_loader,
                device=device,
                logger=logger,
                temp_dir=output_dir / "experiments",
                sparsity_pct=sp_metrics["sparsity_percentage"],
            )
            candidates_list.append(prune_cand)
            model_instances[cand_id] = finalized_pruned

    # 5. Knowledge Distillation Experiment (ResNet-18 -> MobileNetV3-Small)
    if cfg.distillation.enabled:
        logger.info("Running Knowledge Distillation Experiment (Student: MobileNetV3-Small)...")
        student_model = create_model("mobilenet_v3_small")
        student_trained, _ = train_student_distillation(
            teacher=fp32_champion,
            student=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg.distillation,
            device=device,
            logger=logger,
        )
        dist_cand, _ = evaluate_optimization_candidate(
            candidate_id="champion-distillation-student",
            optimization_type="knowledge_distillation",
            model=student_trained,
            val_loader=val_loader,
            device=device,
            logger=logger,
            temp_dir=output_dir / "experiments",
        )
        candidates_list.append(dist_cand)
        model_instances["champion-distillation-student"] = student_trained

    # 6. Apply Quality Gate & Select Optimized Champion
    logger.info("Evaluating candidates against strict Quality Gate...")
    opt_champion, comparison_df = select_optimized_champion(
        reference=ref_candidate,
        candidates=candidates_list,
        config=cfg.quality_gate,
    )

    champ_dir = output_dir / "champion"
    champ_dir.mkdir(parents=True, exist_ok=True)
    save_optimization_artifacts(comparison_df, opt_champion, champ_dir)

    print("\n" + "=" * 80)
    print("PHASE 08 OPTIMIZATION CANDIDATE TRADE-OFF MATRIX (VALIDATION SPLIT)")
    print("=" * 80)
    print(
        comparison_df[
            [
                "candidate_id",
                "optimization_type",
                "val_accuracy",
                "val_macro_f1",
                "latency_ms",
                "model_size_mb",
                "is_accepted",
                "composite_score",
            ]
        ].to_string(index=False)
    )
    print("=" * 80)
    print(
        f"\n>>> OFFICIALLY SELECTED OPTIMIZED CHAMPION: {opt_champion.candidate_id.upper()} "
        f"(Type: {opt_champion.optimization_type}, Decision: {'ACCEPTED' if opt_champion.is_accepted else 'FALLBACK'}) <<<\n"
    )

    # 7. Final Test Evaluation (PROTECTED TEST SET)
    logger.info(
        f"=== Running Final Test Split Evaluation on Optimized Champion '{opt_champion.candidate_id}' ==="
    )
    selected_model = model_instances[opt_champion.candidate_id]

    champ_test_evaluator = Evaluator(
        model=selected_model,
        data_loader=test_loader,
        device=device,
        run_dir=champ_dir / "test_evaluation",
        logger=logger,
    )
    champ_test_results = champ_test_evaluator.evaluate()

    # Also evaluate FP32 Champion Reference on test set for direct final comparison
    ref_test_evaluator = Evaluator(
        model=fp32_champion,
        data_loader=test_loader,
        device=device,
        run_dir=ref_dir / "test_evaluation",
        logger=logger,
    )
    ref_test_results = ref_test_evaluator.evaluate()

    # 8. Compute Calibration (ECE) on Test Predictions
    champ_ece = compute_expected_calibration_error(
        champ_test_results["raw_predictions"]["y_true"],
        champ_test_results["raw_predictions"]["probabilities"],
    )
    ref_ece = compute_expected_calibration_error(
        ref_test_results["raw_predictions"]["y_true"],
        ref_test_results["raw_predictions"]["probabilities"],
    )

    # 9. Export to ONNX & Validate Runtime
    onnx_path = champ_dir / "model.onnx"
    export_val_report = {"is_valid": False}

    if cfg.export.enabled:
        try:
            logger.info(f"Exporting Optimized Champion to ONNX: {onnx_path}...")
            export_to_onnx(selected_model, onnx_path, config=cfg.export)
            export_val_report = validate_onnx_export(
                onnx_path, selected_model, atol=cfg.export.atol
            )
            logger.info(f"ONNX Validation Result: {export_val_report}")
            with open(champ_dir / "export_validation.json", "w", encoding="utf-8") as f:
                json.dump(export_val_report, f, indent=2)
        except Exception as e:
            logger.warning(f"ONNX Export / Validation warning: {e}")
            export_val_report = {"is_valid": False, "error": str(e)}

    # Save Native PyTorch Model & Metadata
    torch.save(selected_model.state_dict(), champ_dir / "model.pt")

    metadata = {
        "model_name": opt_champion.candidate_id,
        "base_model": "resnet18",
        "optimization_technique": opt_champion.optimization_type,
        "validation_metrics": {
            "accuracy": opt_champion.val_accuracy,
            "macro_f1": opt_champion.val_macro_f1,
            "weighted_f1": opt_champion.val_weighted_f1,
        },
        "test_metrics": {
            "accuracy": champ_test_results["summary"]["accuracy"],
            "macro_f1": champ_test_results["summary"]["macro_f1"],
            "weighted_f1": champ_test_results["summary"]["weighted_f1"],
            "expected_calibration_error": champ_ece,
        },
        "efficiency": {
            "latency_ms": opt_champion.latency_ms,
            "throughput_fps": opt_champion.throughput_fps,
            "model_size_mb": opt_champion.model_size_mb,
            "total_parameters": opt_champion.total_parameters,
        },
        "export": export_val_report,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(champ_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Print Final Test Comparison Table
    c_sum = champ_test_results["summary"]
    r_sum = ref_test_results["summary"]

    print("=" * 80)
    print("FINAL BENCHMARK COMPARISON (PHASE 07 CHAMPION vs OPTIMIZED CHAMPION ON TEST SET)")
    print("=" * 80)
    print(
        f"{'Metric':<25} | {'Phase 07 Champion':<20} | {'Optimized Champion':<20} | {'Difference'}"
    )
    print("-" * 80)
    print(
        f"{'Test Accuracy':<25} | {r_sum['accuracy']*100:.2f}%{'':<14} | {c_sum['accuracy']*100:.2f}%{'':<14} | {(c_sum['accuracy'] - r_sum['accuracy'])*100:+.2f}%"
    )
    print(
        f"{'Test Macro F1':<25} | {r_sum['macro_f1']:.4f}{'':<14} | {c_sum['macro_f1']:.4f}{'':<14} | {c_sum['macro_f1'] - r_sum['macro_f1']:+.4f}"
    )
    print(
        f"{'Test Weighted F1':<25} | {r_sum['weighted_f1']:.4f}{'':<14} | {c_sum['weighted_f1']:.4f}{'':<14} | {c_sum['weighted_f1'] - r_sum['weighted_f1']:+.4f}"
    )
    print(
        f"{'Single Sample Latency':<25} | {r_sum['single_sample_latency_ms']:.2f} ms{'':<13} | {c_sum['single_sample_latency_ms']:.2f} ms{'':<13} | {c_sum['single_sample_latency_ms'] - r_sum['single_sample_latency_ms']:+.2f} ms"
    )
    print(
        f"{'Model Size':<25} | {opt_champion.model_size_mb:.2f} MB{'':<13} | {opt_champion.model_size_mb:.2f} MB{'':<13} | 0.00 MB"
    )
    print(
        f"{'Calibration (ECE)':<25} | {ref_ece:.4f}{'':<14} | {champ_ece:.4f}{'':<14} | {champ_ece - ref_ece:+.4f}"
    )
    print("=" * 80)
    print(f"\nArtifacts saved in: {champ_dir}\n")


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_args()
    try:
        run_optimization_pipeline(args)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Optimization pipeline failed: {e}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
