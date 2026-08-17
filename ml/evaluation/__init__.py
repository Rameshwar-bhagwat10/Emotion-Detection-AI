"""Evaluation module exports."""

from __future__ import annotations

from ml.evaluation.benchmark import (
    benchmark_model_inference,
    compute_model_statistics,
    save_benchmark_json,
    save_model_stats_json,
)
from ml.evaluation.classification_report import (
    format_classification_report_table,
    generate_classification_report,
    save_classification_report_csv,
    save_classification_report_json,
)
from ml.evaluation.config import (
    ArtifactsConfig,
    BenchmarkConfig,
    CheckpointEvaluationConfig,
    ConfidenceConfig,
    ErrorAnalysisConfig,
    EvaluationDataConfig,
    EvaluationPipelineConfig,
    ExperimentConfig,
    MetricsEvaluationConfig,
    load_evaluation_config,
)
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
from ml.evaluation.evaluator import Evaluator
from ml.evaluation.metrics import (
    calculate_accuracy,
    calculate_all_metrics,
    calculate_macro_metrics,
    calculate_per_class_metrics,
    calculate_weighted_metrics,
)

__all__ = [
    "ArtifactsConfig",
    "BenchmarkConfig",
    "CheckpointEvaluationConfig",
    "ConfidenceConfig",
    "ErrorAnalysisConfig",
    "EvaluationDataConfig",
    "EvaluationPipelineConfig",
    "Evaluator",
    "ExperimentConfig",
    "MetricsEvaluationConfig",
    "analyze_confidence",
    "benchmark_model_inference",
    "calculate_accuracy",
    "calculate_all_metrics",
    "calculate_macro_metrics",
    "calculate_per_class_metrics",
    "calculate_weighted_metrics",
    "compute_confusion_matrix",
    "compute_model_statistics",
    "compute_normalized_confusion_matrix",
    "extract_confusion_pairs",
    "format_classification_report_table",
    "generate_classification_report",
    "load_evaluation_config",
    "plot_and_save_confidence_distribution",
    "plot_and_save_confusion_matrix",
    "save_benchmark_json",
    "save_classification_report_csv",
    "save_classification_report_json",
    "save_confidence_analysis_json",
    "save_confusion_matrix_csv",
    "save_incorrect_predictions_csv",
    "save_model_stats_json",
]
