"""Model selection engine and multi-criteria decision matrix for transfer learning candidates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelCandidate:
    """Evaluation data and performance profile for a model candidate."""

    model_name: str
    checkpoint_path: str
    val_accuracy: float
    val_macro_f1: float
    val_weighted_f1: float
    val_loss: float
    batch_latency_ms: float
    single_sample_latency_ms: float
    throughput_fps: float
    total_parameters: int
    model_size_mb: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCandidate:
        """Create candidate from dictionary."""
        return cls(
            model_name=str(data["model_name"]),
            checkpoint_path=str(data.get("checkpoint_path", "")),
            val_accuracy=float(data.get("val_accuracy", data.get("accuracy", 0.0))),
            val_macro_f1=float(data.get("val_macro_f1", data.get("macro_f1", 0.0))),
            val_weighted_f1=float(data.get("val_weighted_f1", data.get("weighted_f1", 0.0))),
            val_loss=float(data.get("val_loss", data.get("loss", 0.0))),
            batch_latency_ms=float(data.get("batch_latency_ms", 0.0)),
            single_sample_latency_ms=float(data.get("single_sample_latency_ms", 0.0)),
            throughput_fps=float(
                data.get("throughput_fps", data.get("batch_throughput_samples_per_sec", 0.0))
            ),
            total_parameters=int(data.get("total_parameters", 0)),
            model_size_mb=float(data.get("model_size_mb", data.get("checkpoint_size_mb", 0.0))),
        )


@dataclass
class DecisionWeights:
    """Weights for multi-criteria candidate ranking (must sum to 1.0)."""

    accuracy_weight: float = 0.35
    macro_f1_weight: float = 0.35
    latency_weight: float = 0.15
    size_weight: float = 0.15

    def __post_init__(self) -> None:
        """Validate that weights are non-negative and sum to 1.0."""
        total = self.accuracy_weight + self.macro_f1_weight + self.latency_weight + self.size_weight
        if not np.isclose(total, 1.0, atol=1e-3):
            raise ValueError(f"Decision weights must sum to 1.0, got {total:.4f}")


def compare_candidates(
    candidates: list[ModelCandidate],
    weights: DecisionWeights | None = None,
) -> pd.DataFrame:
    """Score and rank model candidates across accuracy, macro F1, latency, and model size.

    Args:
        candidates: List of ModelCandidate records evaluated on validation split.
        weights: Optional DecisionWeights instance.

    Returns:
        pd.DataFrame sorted by 'composite_score' descending with ranks and metric breakdowns.
    """
    if not candidates:
        raise ValueError("Candidates list cannot be empty")

    w = weights or DecisionWeights()
    records = [asdict(c) for c in candidates]
    df = pd.DataFrame(records)

    # 1. Normalize Accuracy (Higher is better)
    max_acc = df["val_accuracy"].max()
    min_acc = df["val_accuracy"].min()
    acc_norm = (
        (df["val_accuracy"] - min_acc) / (max_acc - min_acc)
        if max_acc > min_acc
        else np.ones(len(df))
    )

    # 2. Normalize Macro F1 (Higher is better)
    max_f1 = df["val_macro_f1"].max()
    min_f1 = df["val_macro_f1"].min()
    f1_norm = (
        (df["val_macro_f1"] - min_f1) / (max_f1 - min_f1) if max_f1 > min_f1 else np.ones(len(df))
    )

    # 3. Normalize Latency (Lower is better -> inverted)
    max_lat = df["single_sample_latency_ms"].max()
    min_lat = df["single_sample_latency_ms"].min()
    lat_norm = (
        (max_lat - df["single_sample_latency_ms"]) / (max_lat - min_lat)
        if max_lat > min_lat
        else np.ones(len(df))
    )

    # 4. Normalize Size (Lower is better -> inverted)
    max_size = df["model_size_mb"].max()
    min_size = df["model_size_mb"].min()
    size_norm = (
        (max_size - df["model_size_mb"]) / (max_size - min_size)
        if max_size > min_size
        else np.ones(len(df))
    )

    # Compute composite decision score
    composite_scores = (
        w.accuracy_weight * acc_norm
        + w.macro_f1_weight * f1_norm
        + w.latency_weight * lat_norm
        + w.size_weight * size_norm
    )

    df["composite_score"] = composite_scores.round(4)
    df = df.sort_values(
        by=["composite_score", "val_macro_f1", "val_accuracy"], ascending=False
    ).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df


def select_champion_model(
    candidates: list[ModelCandidate],
    weights: DecisionWeights | None = None,
) -> tuple[ModelCandidate, pd.DataFrame]:
    """Select the #1 ranked candidate as the official Champion Model.

    Args:
        candidates: List of candidate models.
        weights: Optional decision weights.

    Returns:
        Tuple of (Champion ModelCandidate, full ranking DataFrame).
    """
    ranked_df = compare_candidates(candidates, weights=weights)
    top_row = ranked_df.iloc[0].to_dict()
    champion = ModelCandidate.from_dict(top_row)
    return champion, ranked_df


def save_model_selection_artifacts(
    comparison_df: pd.DataFrame,
    champion: ModelCandidate,
    output_dir: Path,
) -> None:
    """Save selection decision matrix in JSON, CSV, and Markdown formats.

    Args:
        comparison_df: Ranked candidates DataFrame.
        champion: Selected champion candidate.
        output_dir: Directory to save selection artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save CSV
    csv_path = output_dir / "model_selection_matrix.csv"
    comparison_df.to_csv(csv_path, index=False)

    # 2. Save JSON
    json_path = output_dir / "model_selection_matrix.json"
    summary_data = {
        "champion_model": asdict(champion),
        "rankings": comparison_df.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 3. Save Markdown summary
    md_path = output_dir / "model_selection_summary.md"
    md_content = [
        "# Model Selection Decision Matrix",
        "",
        f"**Selected Champion Model:** `{champion.model_name}`",
        f"- **Validation Accuracy:** {champion.val_accuracy * 100:.2f}%",
        f"- **Validation Macro F1:** {champion.val_macro_f1:.4f}",
        f"- **Single Sample Latency:** {champion.single_sample_latency_ms:.2f} ms",
        f"- **Model Size:** {champion.model_size_mb:.2f} MB",
        "",
        "## Candidate Comparison Table",
        "",
        "| Rank | Model Name | Val Acc (%) | Val Macro F1 | Latency (ms) | Size (MB) | Composite Score |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in comparison_df.iterrows():
        md_content.append(
            f"| {int(row['rank'])} | `{row['model_name']}` | {row['val_accuracy']*100:.2f}% | "
            f"{row['val_macro_f1']:.4f} | {row['single_sample_latency_ms']:.2f} | "
            f"{row['model_size_mb']:.2f} | **{row['composite_score']:.4f}** |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content) + "\n")
