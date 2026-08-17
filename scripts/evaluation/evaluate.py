"""CLI entry point for model evaluation and baseline benchmarking on FER2013 test set."""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from torch.utils.data import DataLoader  # noqa: E402

from ml.evaluation.classification_report import format_classification_report_table  # noqa: E402
from ml.evaluation.config import EvaluationPipelineConfig, load_evaluation_config  # noqa: E402
from ml.evaluation.evaluator import Evaluator  # noqa: E402
from ml.models.factory import create_model  # noqa: E402
from ml.preprocessing.dataloaders import build_dataloaders  # noqa: E402
from ml.training.checkpointing import CheckpointManager  # noqa: E402
from ml.utils.device import get_device  # noqa: E402
from ml.utils.logging import setup_training_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate facial emotion detection model on FER2013 test dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ml/configs/evaluation/baseline.yaml",
        help="Path to evaluation YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional override path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name identifier for output directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for evaluation batch size.",
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


def _prepare_dataloader(
    config: EvaluationPipelineConfig,
    split: str,
    batch_size: int,
) -> DataLoader[dict[str, Any]]:
    """Build and return authoritative DataLoader for requested split."""
    raw_data_path = _resolve_path(config.data.raw_path)
    train_loader, val_loader, test_loader = build_dataloaders(
        data_path=raw_data_path,
        batch_size=batch_size,
        num_workers=config.data.num_workers,
    )

    if split == "test":
        return test_loader
    elif split == "val":
        return val_loader
    elif split == "train":
        return train_loader
    else:
        raise ValueError(f"Unknown split: {split}")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Execute evaluation and benchmark pipeline."""
    config_path = _resolve_path(args.config)
    config: EvaluationPipelineConfig = load_evaluation_config(config_path)

    # Setup run dir and logger
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{config.experiment.name}_{timestamp}"
    run_dir = ROOT_DIR / config.artifacts.save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "evaluation.log"
    logger = setup_training_logger(name=f"eval_{run_name}", log_file=log_file)
    logger.info(f"=== Starting Emotion Detection Model Evaluation: {run_name} ===")
    logger.info(f"Loaded evaluation config: {config_path}")

    # Determine device
    device = get_device(args.device)
    logger.info(f"Evaluation compute device: {device.type} ({device})")

    # Resolve checkpoint
    ckpt_path_str = args.checkpoint or config.checkpoint.path
    ckpt_path = _resolve_path(ckpt_path_str)
    logger.info(f"Evaluating checkpoint: {ckpt_path}")

    # Instantiate model via factory
    logger.info(f"Instantiating model '{config.model_name}' via Model Factory...")
    model = create_model(config.model_name)

    # Load weights from checkpoint
    CheckpointManager.load(checkpoint_path=ckpt_path, model=model, device=device)
    logger.info(f"Successfully loaded model weights from {ckpt_path.name}")

    # Build DataLoader for requested split
    target_split = args.split or config.data.split
    batch_size = args.batch_size or config.data.batch_size
    logger.info(f"Loading Phase 03 '{target_split}' DataLoader (batch_size={batch_size})...")
    data_loader = _prepare_dataloader(config, split=target_split, batch_size=batch_size)
    logger.info(f"Loaded {len(data_loader.dataset)} samples across {len(data_loader)} batches.")

    # Initialize Evaluator
    evaluator = Evaluator(
        model=model,
        data_loader=data_loader,
        device=device,
        config=config,
        checkpoint_path=ckpt_path,
        run_dir=run_dir,
        logger=logger,
    )

    # Execute evaluation
    results = evaluator.evaluate()

    # Print summary table to stdout
    table_str = format_classification_report_table(results["classification_report"])
    print("\n" + "=" * 58)
    print(f"CLASSIFICATION REPORT ({config.model_name} on {target_split.upper()} set)")
    print("=" * 58)
    print(table_str)
    print("=" * 58)

    summary = results["summary"]
    print("\nBENCHMARK SUMMARY:")
    print(f"  Overall Accuracy:      {summary['accuracy']*100:.2f}%")
    print(f"  Macro F1-Score:        {summary['macro_f1']:.4f}")
    print(f"  Weighted F1-Score:     {summary['weighted_f1']:.4f}")
    print(f"  Mean Batch Latency:    {summary['batch_latency_ms']:.2f} ms")
    print(f"  Batch Throughput:      {summary['batch_throughput_samples_per_sec']:.2f} samples/sec")
    print(f"  Single Sample Latency: {summary['single_sample_latency_ms']:.2f} ms")
    print(f"  Total Parameters:      {summary['total_parameters']:,}")
    print(f"  Checkpoint Size:       {summary['checkpoint_size_mb']:.2f} MB")
    print(f"  Artifacts directory:   {run_dir}\n")

    return results


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_args()
    try:
        run_evaluation(args)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
