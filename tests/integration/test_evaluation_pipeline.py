"""Integration tests for Phase 06 evaluation and baseline benchmarking pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.evaluation.config import EvaluationPipelineConfig
from ml.evaluation.evaluator import Evaluator
from ml.models.factory import create_model
from ml.preprocessing.dataloaders import build_dataloaders
from ml.training.checkpointing import CheckpointManager

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def create_synthetic_test_dataloader(batch_size: int = 4) -> DataLoader:
    """Create lightweight synthetic test DataLoader."""
    x = torch.randn(16, 1, 48, 48, dtype=torch.float32)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1], dtype=torch.long)
    ds = TensorDataset(x, y)

    def collate_fn(batch):
        return {
            "image": torch.stack([item[0] for item in batch]),
            "label": torch.stack([item[1] for item in batch]),
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def test_evaluator_parameter_immutability(tmp_path: Path) -> None:
    """Verify that evaluation does not mutate model weights under any circumstance."""
    model = create_model("baseline_cnn")
    initial_weights = [p.clone().detach() for p in model.parameters()]

    test_loader = create_synthetic_test_dataloader(batch_size=4)
    evaluator = Evaluator(
        model=model,
        data_loader=test_loader,
        device="cpu",
        run_dir=tmp_path,
    )

    _ = evaluator.evaluate()

    for p_init, p_curr in zip(initial_weights, model.parameters(), strict=True):
        assert torch.equal(p_init, p_curr), "Model weights mutated during evaluation!"


def test_end_to_end_evaluation_artifacts(tmp_path: Path) -> None:
    """Verify all machine-readable and visual artifacts are generated."""
    model = create_model("baseline_cnn")
    test_loader = create_synthetic_test_dataloader(batch_size=4)

    config = EvaluationPipelineConfig()
    evaluator = Evaluator(
        model=model,
        data_loader=test_loader,
        device="cpu",
        config=config,
        run_dir=tmp_path,
    )

    results = evaluator.evaluate()

    assert "summary" in results
    assert "classification_report" in results
    assert "confusion_matrix" in results
    assert "benchmark" in results

    # Verify all files exist
    assert (tmp_path / "evaluation_results.json").exists()
    assert (tmp_path / "classification_report.json").exists()
    assert (tmp_path / "classification_report.csv").exists()
    assert (tmp_path / "confusion_matrix.csv").exists()
    assert (tmp_path / "confusion_matrix_normalized.csv").exists()
    assert (tmp_path / "incorrect_predictions.csv").exists()
    assert (tmp_path / "confusion_pairs.json").exists()
    assert (tmp_path / "confidence_analysis.json").exists()
    assert (tmp_path / "benchmark.json").exists()
    assert (tmp_path / "model_stats.json").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "confidence_distribution.png").exists()


def test_real_dataset_and_checkpoint_evaluation(tmp_path: Path) -> None:
    """Verify evaluation on actual Phase 03 test DataLoader and Phase 05 checkpoint."""
    raw_csv = ROOT_DIR / "data/raw/fer2013/fer2013.csv"
    if not raw_csv.exists():
        pytest.skip("FER2013 dataset not present on host")

    _, _, test_loader = build_dataloaders(raw_csv, batch_size=64)

    model = create_model("baseline_cnn")

    # Find any available trained checkpoint from Phase 05
    ckpt_dir = ROOT_DIR / "artifacts/training/baseline_cnn"
    ckpt_files = list(ckpt_dir.glob("**/best.pt"))
    if not ckpt_files:
        pytest.skip("No trained checkpoint found")

    ckpt_path = ckpt_files[0]
    CheckpointManager.load(ckpt_path, model=model, device=torch.device("cpu"))

    evaluator = Evaluator(
        model=model,
        data_loader=test_loader,
        device="cpu",
        checkpoint_path=ckpt_path,
        run_dir=tmp_path,
    )

    results = evaluator.evaluate()
    assert results["summary"]["total_samples"] == len(test_loader.dataset)
    assert 0.0 <= results["summary"]["accuracy"] <= 1.0
    assert (tmp_path / "evaluation_results.json").exists()
