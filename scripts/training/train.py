"""CLI entry point for model training and experimentation."""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ml.models.factory import create_model  # noqa: E402
from ml.preprocessing.dataloaders import build_dataloaders  # noqa: E402
from ml.training.checkpointing import CheckpointManager  # noqa: E402
from ml.training.config import TrainingPipelineConfig, load_training_config  # noqa: E402
from ml.training.early_stopping import EarlyStopping  # noqa: E402
from ml.training.losses import create_loss  # noqa: E402
from ml.training.optimizers import create_optimizer  # noqa: E402
from ml.training.schedulers import create_scheduler  # noqa: E402
from ml.training.trainer import Trainer  # noqa: E402
from ml.utils.device import get_device  # noqa: E402
from ml.utils.logging import setup_training_logger  # noqa: E402
from ml.utils.reproducibility import setup_reproducibility  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train facial emotion detection models on FER2013 dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ml/configs/training/baseline.yaml",
        help="Path to training configuration YAML file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Hardware compute device to execute training on.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for total training epochs.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast single-epoch smoke test to verify pipeline execution.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name identifier. If omitted, uses timestamp.",
    )
    return parser.parse_args()


def _resolve_config_path(raw_path_str: str) -> Path:
    """Locate and return valid config path."""
    p = Path(raw_path_str)
    if p.exists():
        return p
    alt = ROOT_DIR / raw_path_str
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Configuration file not found: {raw_path_str}")


def _setup_run_environment(
    config: TrainingPipelineConfig,
    args: argparse.Namespace,
) -> tuple[Path, logging.Logger, torch.device]:
    """Initialize run directory, logger, and reproducibility."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{config.experiment.name}_{timestamp}"
    if args.smoke_test and not args.run_name:
        run_name = f"smoke_test_{timestamp}"

    run_dir = ROOT_DIR / config.training.checkpoint.save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "training.log"
    logger = setup_training_logger(name=f"train_{run_name}", log_file=log_file)
    logger.info(f"=== Starting Emotion Detection Training Run: {run_name} ===")

    setup_reproducibility(
        seed=config.reproducibility.seed,
        deterministic=config.reproducibility.deterministic,
    )

    device = get_device(args.device)
    logger.info(f"Compute device selected: {device.type} ({device})")
    return run_dir, logger, device


def _prepare_dataloaders(
    config: TrainingPipelineConfig,
    logger: logging.Logger,
) -> tuple[DataLoader, DataLoader]:
    """Build and return train and validation DataLoaders."""
    raw_data_path = ROOT_DIR / config.data.raw_path
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {raw_data_path}")

    batch_size = config.training.batch_size
    train_loader, val_loader, _ = build_dataloaders(
        data_path=raw_data_path,
        batch_size=batch_size,
        num_workers=config.data.num_workers,
    )
    logger.info(
        f"DataLoaders ready: Train={len(train_loader)} batches, Val={len(val_loader)} batches"
    )
    return train_loader, val_loader


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Execute training pipeline given CLI arguments."""
    config_path = _resolve_config_path(args.config)
    config: TrainingPipelineConfig = load_training_config(config_path)

    run_dir, logger, device = _setup_run_environment(config, args)
    train_loader, val_loader = _prepare_dataloaders(config, logger)

    logger.info(f"Instantiating model '{config.model_name}' via Model Factory...")
    model = create_model(config.model_name)

    criterion = create_loss(config.training.loss)
    optimizer = create_optimizer(model.parameters(), config.training.optimizer)
    scheduler, is_metric_dep = create_scheduler(optimizer, config.training.scheduler)

    checkpoint_manager = CheckpointManager.from_config(config.training.checkpoint, run_dir=run_dir)
    early_stopping = (
        EarlyStopping.from_config(config.training.early_stopping)
        if config.training.early_stopping.enabled
        else None
    )

    start_epoch = 1
    loaded_checkpoint: dict[str, Any] | None = None
    if args.resume:
        resume_path = Path(args.resume) if Path(args.resume).exists() else ROOT_DIR / args.resume
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint file not found: {args.resume}")

        logger.info(f"Resuming training from checkpoint: {resume_path}")
        loaded_checkpoint = CheckpointManager.load(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = int(loaded_checkpoint.get("epoch", 0)) + 1
        if "best_metric" in loaded_checkpoint:
            checkpoint_manager.best_metric = float(loaded_checkpoint["best_metric"])
        if "best_epoch" in loaded_checkpoint:
            checkpoint_manager.best_epoch = int(loaded_checkpoint["best_epoch"])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        is_scheduler_metric_dependent=is_metric_dep,
        device=device,
        config=config,
        checkpoint_manager=checkpoint_manager,
        early_stopping=early_stopping,
        run_dir=run_dir,
        use_amp=config.training.mixed_precision,
        gradient_clipping=config.training.gradient_clipping,
        max_norm=config.training.max_norm,
        custom_logger=logger,
    )

    if loaded_checkpoint and "history" in loaded_checkpoint:
        trainer.history = list(loaded_checkpoint["history"])

    total_epochs = 1 if args.smoke_test else (args.epochs or config.training.epochs)
    if start_epoch > total_epochs:
        logger.info(f"Checkpoint epoch ({start_epoch - 1}) >= target epochs ({total_epochs}).")
        return {"status": "already_completed", "run_dir": str(run_dir)}

    results = trainer.fit(epochs=total_epochs, start_epoch=start_epoch)
    logger.info(f"Training completed successfully. Artifacts saved to: {run_dir}")
    return results


def main() -> None:
    """Main CLI execution."""
    args = parse_args()
    try:
        run_training(args)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Training run failed: {e}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
