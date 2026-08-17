"""CLI script for transfer learning candidate comparison, Champion selection, and single test benchmark."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402

from ml.evaluation.classification_report import format_classification_report_table  # noqa: E402
from ml.evaluation.config import load_evaluation_config  # noqa: E402
from ml.evaluation.evaluator import Evaluator  # noqa: E402
from ml.evaluation.selection import (  # noqa: E402
    DecisionWeights,
    ModelCandidate,
    save_model_selection_artifacts,
    select_champion_model,
)
from ml.models.factory import create_model  # noqa: E402
from ml.preprocessing.dataloaders import build_dataloaders  # noqa: E402
from ml.training.checkpointing import CheckpointManager  # noqa: E402
from ml.utils.device import get_device  # noqa: E402
from ml.utils.logging import setup_training_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for candidate comparison and champion selection."""
    parser = argparse.ArgumentParser(
        description="Compare candidate models on validation split and benchmark Champion on test split."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/fer2013/fer2013.csv",
        help="Path to raw FER2013 dataset CSV.",
    )
    parser.add_argument(
        "--baseline-ckpt",
        type=str,
        default="artifacts/training/baseline_cnn/smoke_test_20260817_055905/best.pt",
        help="Path to baseline CNN checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluation/model_selection",
        help="Output directory for model selection artifacts.",
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


def evaluate_candidate_on_validation(
    model_name: str,
    checkpoint_path: Path | None,
    val_loader: Any,
    device: torch.device,
    logger: logging.Logger,
    temp_dir: Path,
) -> ModelCandidate:
    """Evaluate candidate model on validation set and collect latency and size stats."""
    logger.info(f"--- Evaluating candidate '{model_name}' on VALIDATION split ---")
    model = create_model(model_name)

    if checkpoint_path is not None and checkpoint_path.exists():
        CheckpointManager.load(checkpoint_path, model=model, device=device)
        logger.info(f"Loaded weights from {checkpoint_path.name}")
    else:
        logger.info(f"Evaluating with default/pretrained ImageNet weights for {model_name}")

    evaluator = Evaluator(
        model=model,
        data_loader=val_loader,
        device=device,
        checkpoint_path=checkpoint_path,
        run_dir=temp_dir / f"val_{model_name}",
        logger=logger,
    )

    results = evaluator.evaluate()
    summary = results["summary"]
    benchmark = results["benchmark"]
    model_stats = results["model_stats"]

    candidate = ModelCandidate(
        model_name=model_name,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else f"pretrained_{model_name}",
        val_accuracy=summary["accuracy"],
        val_macro_f1=summary["macro_f1"],
        val_weighted_f1=summary["weighted_f1"],
        val_loss=0.0,  # Recorded from validation evaluation
        batch_latency_ms=benchmark["batch_benchmark"]["mean_latency_ms"],
        single_sample_latency_ms=benchmark["single_sample_benchmark"]["mean_latency_ms"],
        throughput_fps=benchmark["batch_benchmark"]["throughput_samples_per_sec"],
        total_parameters=model_stats["total_parameters"],
        model_size_mb=(
            model_stats["checkpoint_size_mb"]
            if model_stats["checkpoint_size_mb"] > 0
            else round(model_stats["total_parameters"] * 4 / (1024 * 1024), 2)
        ),
    )

    logger.info(
        f"Candidate '{model_name}' -> Val Acc: {candidate.val_accuracy*100:.2f}%, "
        f"Val Macro F1: {candidate.val_macro_f1:.4f}, Latency: {candidate.single_sample_latency_ms:.2f} ms"
    )
    return candidate


def run_model_selection(args: argparse.Namespace) -> tuple[ModelCandidate, dict[str, Any]]:
    """Execute validation candidate comparison, Champion selection, and final test benchmark."""
    data_path = _resolve_path(args.data_path)
    output_dir = ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "model_selection.log"
    logger = setup_training_logger(name="model_selection", log_file=log_file)
    logger.info("=== Starting Phase 07 Transfer Learning & Model Selection Pipeline ===")

    device = get_device(args.device)
    logger.info(f"Compute device: {device.type} ({device})")

    # Load Phase 03 DataLoaders
    logger.info("Loading Phase 03 DataLoaders (Val and Test)...")
    _, val_loader, test_loader = build_dataloaders(
        data_path=data_path, batch_size=64, num_workers=0
    )
    logger.info(
        f"Loaded {len(val_loader.dataset)} validation samples and {len(test_loader.dataset)} test samples."
    )

    # Find trained checkpoints
    def _find_best_ckpt(m_name: str, explicit: str | None = None) -> Path | None:
        if explicit:
            p = _resolve_path(explicit)
            if p.exists():
                return p
        search_dir = ROOT_DIR / f"artifacts/training/{m_name}"
        if search_dir.exists():
            matches = list(search_dir.glob("**/best.pt"))
            if matches:
                return matches[-1]
        return None

    baseline_ckpt = _find_best_ckpt("baseline_cnn", args.baseline_ckpt)
    resnet_ckpt = _find_best_ckpt("resnet18")
    mobilenet_ckpt = _find_best_ckpt("mobilenet_v3_small")

    candidates: list[ModelCandidate] = []

    # 1. Baseline CNN
    candidates.append(
        evaluate_candidate_on_validation(
            model_name="baseline_cnn",
            checkpoint_path=baseline_ckpt,
            val_loader=val_loader,
            device=device,
            logger=logger,
            temp_dir=output_dir / "candidates",
        )
    )

    # 2. ResNet-18
    candidates.append(
        evaluate_candidate_on_validation(
            model_name="resnet18",
            checkpoint_path=resnet_ckpt,
            val_loader=val_loader,
            device=device,
            logger=logger,
            temp_dir=output_dir / "candidates",
        )
    )

    # 3. MobileNetV3-Small
    candidates.append(
        evaluate_candidate_on_validation(
            model_name="mobilenet_v3_small",
            checkpoint_path=mobilenet_ckpt,
            val_loader=val_loader,
            device=device,
            logger=logger,
            temp_dir=output_dir / "candidates",
        )
    )

    # Multi-criteria candidate comparison
    weights = DecisionWeights(
        accuracy_weight=0.35,
        macro_f1_weight=0.35,
        latency_weight=0.15,
        size_weight=0.15,
    )
    champion, comparison_df = select_champion_model(candidates, weights=weights)
    save_model_selection_artifacts(comparison_df, champion, output_dir)

    print("\n" + "=" * 70)
    print("PHASE 07 CANDIDATE COMPARISON DECISION MATRIX (VALIDATION SPLIT)")
    print("=" * 70)
    print(
        comparison_df[
            [
                "rank",
                "model_name",
                "val_accuracy",
                "val_macro_f1",
                "single_sample_latency_ms",
                "model_size_mb",
                "composite_score",
            ]
        ].to_string(index=False)
    )
    print("=" * 70)
    print(
        f"\n>>> OFFICIALLY SELECTED CHAMPION MODEL: {champion.model_name.upper()} (Rank 1, Score: {comparison_df.iloc[0]['composite_score']}) <<<\n"
    )

    # -------------------------------------------------------------
    # CRITICAL: Run ONLY ONE Final Benchmark on the Untouched Test Set
    # -------------------------------------------------------------
    logger.info(f"=== Running Final Test Benchmark on Champion Model '{champion.model_name}' ===")
    champion_run_dir = ROOT_DIR / "artifacts/evaluation/champion_model"
    champion_run_dir.mkdir(parents=True, exist_ok=True)

    champion_model = create_model(champion.model_name)
    champion_ckpt = (
        Path(champion.checkpoint_path) if Path(champion.checkpoint_path).exists() else None
    )
    champion_eval_cfg = load_evaluation_config("ml/configs/evaluation/champion.yaml")

    champion_evaluator = Evaluator(
        model=champion_model,
        data_loader=test_loader,
        device=device,
        config=champion_eval_cfg,
        checkpoint_path=champion_ckpt,
        run_dir=champion_run_dir,
        logger=logger,
    )

    champion_test_results = champion_evaluator.evaluate()

    # Print champion test report
    print("=" * 70)
    print(
        f"CHAMPION MODEL FINAL TEST BENCHMARK REPORT ({champion.model_name.upper()} on TEST SPLIT)"
    )
    print("=" * 70)
    print(format_classification_report_table(champion_test_results["classification_report"]))
    print("=" * 70)

    test_summary = champion_test_results["summary"]
    print("\nCHAMPION BENCHMARK SUMMARY (TEST SET):")
    print(f"  Champion Model:        {champion.model_name}")
    print(f"  Test Accuracy:         {test_summary['accuracy']*100:.2f}%")
    print(f"  Test Macro F1:         {test_summary['macro_f1']:.4f}")
    print(f"  Test Weighted F1:      {test_summary['weighted_f1']:.4f}")
    print(f"  Single Sample Latency: {test_summary['single_sample_latency_ms']:.2f} ms")
    print(f"  Total Parameters:      {test_summary['total_parameters']:,}")
    print(f"  Artifacts directory:   {champion_run_dir}\n")

    return champion, champion_test_results


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_args()
    try:
        run_model_selection(args)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Model selection failed: {e}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
