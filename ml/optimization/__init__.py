"""Model optimization and deployment readiness module exports."""

from __future__ import annotations

from ml.optimization.config import (
    DistillationConfig,
    ExportConfig,
    FP16Config,
    OptimizationPipelineConfig,
    PruningConfig,
    QualityGateConfig,
    QuantizationConfig,
    load_optimization_config,
)
from ml.optimization.distillation import DistillationLoss, train_student_distillation
from ml.optimization.exporter import export_to_onnx, validate_onnx_export
from ml.optimization.precision import ReducedPrecisionWrapper, evaluate_numerical_difference
from ml.optimization.pruning import (
    apply_magnitude_pruning,
    compute_model_sparsity,
    finalize_pruning,
    fine_tune_pruned_model,
)
from ml.optimization.quantization import calibrate_static_ptq, quantize_model_dynamic
from ml.optimization.selection import (
    OptimizationCandidate,
    evaluate_quality_gate,
    save_optimization_artifacts,
    select_optimized_champion,
)

__all__ = [
    "DistillationConfig",
    "DistillationLoss",
    "ExportConfig",
    "FP16Config",
    "OptimizationCandidate",
    "OptimizationPipelineConfig",
    "PruningConfig",
    "QualityGateConfig",
    "QuantizationConfig",
    "ReducedPrecisionWrapper",
    "apply_magnitude_pruning",
    "calibrate_static_ptq",
    "compute_model_sparsity",
    "evaluate_numerical_difference",
    "evaluate_quality_gate",
    "export_to_onnx",
    "finalize_pruning",
    "fine_tune_pruned_model",
    "load_optimization_config",
    "quantize_model_dynamic",
    "save_optimization_artifacts",
    "select_optimized_champion",
    "train_student_distillation",
    "validate_onnx_export",
]
