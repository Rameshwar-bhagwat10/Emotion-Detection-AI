"""Quality Gate, candidate comparison, and Optimized Champion selection engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.optimization.config import QualityGateConfig


@dataclass
class OptimizationCandidate:
    """Performance and efficiency profile for an optimization candidate."""

    candidate_id: str
    base_model: str
    optimization_type: str
    val_accuracy: float
    val_macro_f1: float
    val_weighted_f1: float
    latency_ms: float
    throughput_fps: float
    model_size_mb: float
    total_parameters: int
    sparsity_pct: float = 0.0
    is_accepted: bool = False
    rejection_reason: str = ""
    composite_score: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationCandidate:
        """Create candidate from dictionary."""
        return cls(
            candidate_id=str(data["candidate_id"]),
            base_model=str(data.get("base_model", "resnet18")),
            optimization_type=str(data.get("optimization_type", "fp32_reference")),
            val_accuracy=float(data.get("val_accuracy", 0.0)),
            val_macro_f1=float(data.get("val_macro_f1", 0.0)),
            val_weighted_f1=float(data.get("val_weighted_f1", 0.0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            throughput_fps=float(data.get("throughput_fps", 0.0)),
            model_size_mb=float(data.get("model_size_mb", 0.0)),
            total_parameters=int(data.get("total_parameters", 0)),
            sparsity_pct=float(data.get("sparsity_pct", 0.0)),
            is_accepted=bool(data.get("is_accepted", False)),
            rejection_reason=str(data.get("rejection_reason", "")),
            composite_score=float(data.get("composite_score", 0.0)),
        )


def evaluate_quality_gate(
    reference: OptimizationCandidate,
    candidate: OptimizationCandidate,
    config: QualityGateConfig | None = None,
) -> tuple[bool, str]:
    """Check whether candidate satisfies strict quality gate thresholds relative to reference.

    Args:
        reference: FP32 Champion Reference candidate.
        candidate: Candidate under evaluation.
        config: Optional QualityGateConfig.

    Returns:
        Tuple of (is_accepted, reason_string).
    """
    if candidate.candidate_id == reference.candidate_id:
        return True, "FP32 Champion Reference Baseline"

    cfg = config or QualityGateConfig()

    macro_f1_drop = reference.val_macro_f1 - candidate.val_macro_f1
    if macro_f1_drop > cfg.max_macro_f1_drop:
        return (
            False,
            f"Macro F1 drop {macro_f1_drop:.4f} exceeds max allowed threshold {cfg.max_macro_f1_drop:.4f}",
        )

    acc_drop = reference.val_accuracy - candidate.val_accuracy
    if acc_drop > cfg.max_accuracy_drop:
        return (
            False,
            f"Accuracy drop {acc_drop:.4f} exceeds max allowed threshold {cfg.max_accuracy_drop:.4f}",
        )

    # Check for efficiency gain
    latency_reduction_pct = (
        (reference.latency_ms - candidate.latency_ms) / reference.latency_ms * 100.0
        if reference.latency_ms > 0
        else 0.0
    )
    size_reduction_pct = (
        (reference.model_size_mb - candidate.model_size_mb) / reference.model_size_mb * 100.0
        if reference.model_size_mb > 0
        else 0.0
    )

    if (
        latency_reduction_pct < cfg.min_latency_reduction_pct
        and size_reduction_pct < cfg.min_size_reduction_pct
    ):
        return (
            False,
            f"Efficiency gain insufficient (latency reduction: {latency_reduction_pct:.1f}%, size reduction: {size_reduction_pct:.1f}%)",
        )

    return True, "Passed all quality and efficiency thresholds"


def select_optimized_champion(
    reference: OptimizationCandidate,
    candidates: list[OptimizationCandidate],
    config: QualityGateConfig | None = None,
) -> tuple[OptimizationCandidate, pd.DataFrame]:
    """Score candidates, apply quality gate, and select Optimized Champion with fallback guarantee.

    Args:
        reference: Reference FP32 Champion profile.
        candidates: List of optimization candidate profiles evaluated on validation split.
        config: QualityGateConfig instance.

    Returns:
        Tuple of (Selected Optimized Champion, complete ranked DataFrame).
    """
    cfg = config or QualityGateConfig()
    all_candidates = [reference] + [
        c for c in candidates if c.candidate_id != reference.candidate_id
    ]

    records = []
    for cand in all_candidates:
        accepted, reason = evaluate_quality_gate(reference, cand, cfg)
        cand.is_accepted = accepted
        cand.rejection_reason = reason
        records.append(asdict(cand))

    df = pd.DataFrame(records)

    # Compute composite efficiency-quality score for ranking
    max_f1 = df["val_macro_f1"].max()
    min_f1 = df["val_macro_f1"].min()
    f1_norm = (
        (df["val_macro_f1"] - min_f1) / (max_f1 - min_f1) if max_f1 > min_f1 else np.ones(len(df))
    )

    max_size = df["model_size_mb"].max()
    min_size = df["model_size_mb"].min()
    size_norm = (
        (max_size - df["model_size_mb"]) / (max_size - min_size)
        if max_size > min_size
        else np.ones(len(df))
    )

    max_lat = df["latency_ms"].max()
    min_lat = df["latency_ms"].min()
    lat_norm = (
        (max_lat - df["latency_ms"]) / (max_lat - min_lat)
        if max_lat > min_lat
        else np.ones(len(df))
    )

    df["composite_score"] = (0.50 * f1_norm + 0.25 * size_norm + 0.25 * lat_norm).round(4)

    # Rank accepted candidates first, sorted by composite_score
    accepted_df = df[df["is_accepted"]].sort_values(
        by=["composite_score", "val_macro_f1"], ascending=False
    )

    if len(accepted_df) > 0:
        top_row = accepted_df.iloc[0].to_dict()
        champion = OptimizationCandidate.from_dict(top_row)
    else:
        # Fallback to FP32 reference
        champion = reference

    df["rank"] = (
        df.sort_values(by=["is_accepted", "composite_score"], ascending=[False, False])
        .reset_index()
        .index
        + 1
    )
    return champion, df


def save_optimization_artifacts(
    comparison_df: pd.DataFrame,
    champion: OptimizationCandidate,
    output_dir: Path,
) -> None:
    """Save optimization comparison matrix in CSV, JSON, and Markdown formats.

    Args:
        comparison_df: Complete candidates comparison DataFrame.
        champion: Selected Optimized Champion.
        output_dir: Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save CSV
    csv_path = output_dir / "optimization_matrix.csv"
    comparison_df.to_csv(csv_path, index=False)

    # 2. Save JSON
    json_path = output_dir / "optimization_matrix.json"
    summary_data = {
        "optimized_champion": asdict(champion),
        "candidates": comparison_df.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 3. Save Markdown Report
    md_path = output_dir / "selection_report.md"
    lines = [
        "# Model Optimization & Candidate Selection Report",
        "",
        f"**Selected Optimized Champion:** `{champion.candidate_id}` (`{champion.optimization_type}`)",
        f"- **Validation Accuracy:** {champion.val_accuracy * 100:.2f}%",
        f"- **Validation Macro F1:** {champion.val_macro_f1:.4f}",
        f"- **Inference Latency:** {champion.latency_ms:.2f} ms",
        f"- **Model Size:** {champion.model_size_mb:.2f} MB",
        f"- **Acceptance Decision:** `{'ACCEPTED' if champion.is_accepted else 'FALLBACK'}` ({champion.rejection_reason})",
        "",
        "## Optimization Candidate Trade-off Matrix",
        "",
        "| Candidate ID | Optimization | Val Acc (%) | Val Macro F1 | Latency (ms) | Size (MB) | Accepted | Decision Reason |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for _, row in comparison_df.iterrows():
        status = "YES" if row["is_accepted"] else "NO"
        lines.append(
            f"| `{row['candidate_id']}` | `{row['optimization_type']}` | {row['val_accuracy']*100:.2f}% | "
            f"{row['val_macro_f1']:.4f} | {row['latency_ms']:.2f} | {row['model_size_mb']:.2f} | "
            f"**{status}** | {row['rejection_reason']} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
