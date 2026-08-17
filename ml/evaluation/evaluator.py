"""Model evaluation engine orchestrating test inference, metrics, error analysis, and artifact generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.datasets.fer2013.parser import EMOTION_NAMES
from ml.evaluation.benchmark import (
    benchmark_model_inference,
    compute_model_statistics,
    save_benchmark_json,
    save_model_stats_json,
)
from ml.evaluation.classification_report import (
    generate_classification_report,
    save_classification_report_csv,
    save_classification_report_json,
)
from ml.evaluation.config import EvaluationPipelineConfig
from ml.evaluation.confusion_matrix import (
    compute_confusion_matrix,
    compute_normalized_confusion_matrix,
    extract_confusion_pairs,
    plot_and_save_confusion_matrix,
    save_confusion_matrix_csv,
)
from ml.evaluation.error_analysis import (
    analyze_confidence,
    plot_and_save_confidence_distribution,
    save_confidence_analysis_json,
    save_incorrect_predictions_csv,
)


class Evaluator:
    """Evaluator engine executing inference and multi-dimensional analysis on test datasets."""

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader[dict[str, Any]],
        device: torch.device | str = "cpu",
        config: EvaluationPipelineConfig | None = None,
        class_names: list[str] | None = None,
        checkpoint_path: Path | str | None = None,
        run_dir: Path | str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize Evaluator.

        Args:
            model: PyTorch model to evaluate.
            data_loader: Evaluation data loader (e.g. test_loader).
            device: Compute device.
            config: Optional evaluation configuration.
            class_names: Canonical class names list (defaults to Phase 02 EMOTION_NAMES).
            checkpoint_path: Optional path to evaluated checkpoint file.
            run_dir: Output directory for evaluation artifacts.
            logger: Optional logger instance.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.data_loader = data_loader
        self.config = config or EvaluationPipelineConfig()
        self.class_names = class_names or list(EMOTION_NAMES)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.run_dir = (
            Path(run_dir) if run_dir else Path("artifacts/evaluation/baseline_cnn/default_eval")
        )
        self.logger = logger or logging.getLogger("Evaluator")

    def collect_predictions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute model forward pass over DataLoader and collect predictions.

        Returns:
            Tuple of (y_true, y_pred, probabilities) as NumPy arrays on CPU.
        """
        self.model.eval()

        all_y_true: list[int] = []
        all_y_pred: list[int] = []
        all_probs: list[np.ndarray] = []

        softmax = nn.Softmax(dim=-1)

        with torch.no_grad():
            for batch in self.data_loader:
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                logits = self.model(images)
                probs = softmax(logits)
                preds = torch.argmax(logits, dim=-1)

                all_y_true.extend(labels.cpu().numpy().tolist())
                all_y_pred.extend(preds.cpu().numpy().tolist())
                all_probs.append(probs.cpu().numpy())

        y_true_arr = np.array(all_y_true, dtype=np.int64)
        y_pred_arr = np.array(all_y_pred, dtype=np.int64)
        probs_arr = np.vstack(all_probs) if all_probs else np.empty((0, len(self.class_names)))

        return y_true_arr, y_pred_arr, probs_arr

    def evaluate(self) -> dict[str, Any]:
        """Execute full evaluation pipeline and generate all artifacts.

        Returns:
            Comprehensive evaluation results dictionary.
        """
        dataset_len = (
            len(self.data_loader.dataset)
            if hasattr(self.data_loader.dataset, "__len__")
            else "unknown"
        )
        self.logger.info(f"Starting evaluation run on {dataset_len} samples...")

        # 1. Collect predictions
        y_true, y_pred, probs = self.collect_predictions()
        self.logger.info(f"Predictions collected for {len(y_true)} samples.")

        # 2. Classification Report & Core Metrics
        report = generate_classification_report(y_true, y_pred, self.class_names)
        self.logger.info(
            f"Overall Accuracy: {report['accuracy']*100:.2f}% | "
            f"Macro F1: {report['macro_avg']['f1-score']:.4f} | "
            f"Weighted F1: {report['weighted_avg']['f1-score']:.4f}"
        )

        save_classification_report_json(report, self.run_dir / "classification_report.json")
        save_classification_report_csv(report, self.run_dir / "classification_report.csv")

        # 3. Confusion Matrix
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(self.class_names))
        norm_cm = compute_normalized_confusion_matrix(cm)
        save_confusion_matrix_csv(cm, self.class_names, self.run_dir / "confusion_matrix.csv")
        save_confusion_matrix_csv(
            norm_cm, self.class_names, self.run_dir / "confusion_matrix_normalized.csv"
        )

        confusion_pairs = extract_confusion_pairs(cm, self.class_names)
        with open(self.run_dir / "confusion_pairs.json", "w", encoding="utf-8") as f:
            json.dump(confusion_pairs, f, indent=2)

        if self.config.artifacts.save_visualizations:
            plot_and_save_confusion_matrix(
                cm, self.class_names, self.run_dir / "confusion_matrix.png", normalized=False
            )
            plot_and_save_confusion_matrix(
                norm_cm,
                self.class_names,
                self.run_dir / "confusion_matrix_normalized.png",
                normalized=True,
            )

        # 4. Confidence and Error Analysis
        conf_analysis = analyze_confidence(
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probs,
            class_names=self.class_names,
            high_confidence_threshold=self.config.confidence.high_confidence_threshold,
            num_bins=self.config.confidence.num_bins,
        )
        save_confidence_analysis_json(conf_analysis, self.run_dir / "confidence_analysis.json")

        if self.config.error_analysis.save_incorrect:
            save_incorrect_predictions_csv(
                conf_analysis.get("all_incorrect_samples", []),
                self.run_dir / "incorrect_predictions.csv",
            )

        if self.config.artifacts.save_visualizations:
            correct_mask = y_true == y_pred
            confs = np.max(probs, axis=-1)
            plot_and_save_confidence_distribution(
                correct_confs=confs[correct_mask],
                incorrect_confs=confs[~correct_mask],
                output_path=self.run_dir / "confidence_distribution.png",
                num_bins=self.config.confidence.num_bins,
            )

        # 5. Benchmarking & Model Statistics
        self.logger.info("Executing inference latency and throughput benchmarking...")
        benchmark_results = benchmark_model_inference(
            model=self.model,
            device=self.device,
            input_shape=(1, 48, 48),
            batch_size=self.config.benchmark.batch_size,
            warmup_iterations=self.config.benchmark.warmup_iterations,
            benchmark_iterations=self.config.benchmark.benchmark_iterations,
        )
        save_benchmark_json(benchmark_results, self.run_dir / "benchmark.json")

        model_stats = compute_model_statistics(
            model=self.model,
            checkpoint_path=self.checkpoint_path,
        )
        save_model_stats_json(model_stats, self.run_dir / "model_stats.json")

        # 6. Top-level Summary JSON
        summary: dict[str, Any] = {
            "experiment": self.config.experiment.name,
            "model": self.config.model_name,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else "None",
            "split": self.config.data.split,
            "total_samples": len(y_true),
            "accuracy": report["accuracy"],
            "macro_precision": report["macro_avg"]["precision"],
            "macro_recall": report["macro_avg"]["recall"],
            "macro_f1": report["macro_avg"]["f1-score"],
            "weighted_precision": report["weighted_avg"]["precision"],
            "weighted_recall": report["weighted_avg"]["recall"],
            "weighted_f1": report["weighted_avg"]["f1-score"],
            "high_confidence_error_count": conf_analysis["high_confidence_errors"]["count"],
            "mean_confidence": conf_analysis["overall"]["mean_confidence"],
            "batch_latency_ms": benchmark_results["batch_benchmark"]["mean_latency_ms"],
            "batch_throughput_samples_per_sec": benchmark_results["batch_benchmark"][
                "throughput_samples_per_sec"
            ],
            "single_sample_latency_ms": benchmark_results["single_sample_benchmark"][
                "mean_latency_ms"
            ],
            "total_parameters": model_stats["total_parameters"],
            "checkpoint_size_mb": model_stats["checkpoint_size_mb"],
        }

        with open(self.run_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Evaluation complete. All artifacts saved to {self.run_dir}")
        return {
            "summary": summary,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "confusion_pairs": confusion_pairs,
            "confidence_analysis": {
                k: v for k, v in conf_analysis.items() if k != "all_incorrect_samples"
            },
            "benchmark": benchmark_results,
            "model_stats": model_stats,
            "run_dir": str(self.run_dir),
        }
