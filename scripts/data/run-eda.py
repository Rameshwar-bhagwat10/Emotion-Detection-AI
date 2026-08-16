"""Run complete Exploratory Data Analysis (EDA) on FER2013.

Orchestrates loading, validation, statistical analysis, duplicate/leakage detection,
chart generation, sample visualization grids, and JSON reporting.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add root directory to pythonpath
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.datasets.fer2013.loader import FER2013Loader  # noqa: E402
from ml.datasets.fer2013.parser import EMOTION_NAMES  # noqa: E402
from ml.datasets.fer2013.statistics import FER2013Statistics  # noqa: E402
from ml.datasets.fer2013.validator import FER2013Validator  # noqa: E402
from ml.utils.seed import set_seed  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run-eda")


def run_eda(
    data_path: str = "data/raw/fer2013/fer2013.csv",
    output_dir: str = "data/interim/fer2013/eda",
    seed: int = 42,
) -> None:
    """Execute complete EDA pipeline."""
    set_seed(seed)
    data_file = ROOT_DIR / data_path
    eda_dir = ROOT_DIR / output_dir

    charts_dir = eda_dir / "charts"
    samples_dir = eda_dir / "samples"
    reports_dir = eda_dir / "reports"

    for d in [charts_dir, samples_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("==================================================")
    logger.info("STARTING FER2013 EXPLORATORY DATA ANALYSIS (EDA)")
    logger.info("==================================================")

    # 1. Validation
    logger.info("Step 1/5: Validating raw dataset integrity...")
    validator = FER2013Validator()
    val_result = validator.validate_dataset(data_file)
    val_report_file = reports_dir / "validation_report.json"
    with open(val_report_file, "w", encoding="utf-8") as f:
        f.write(val_result.to_json(indent=2))
    logger.info("Saved validation report to: %s", val_report_file)

    if not val_result.is_valid and val_result.valid_records == 0:
        logger.error("Dataset validation failed: No valid records found.")
        sys.exit(1)

    # 2. Loading Records
    logger.info("Step 2/5: Loading dataset records...")
    loader = FER2013Loader(data_file)
    records = loader.load_records()
    logger.info("Successfully loaded %d records into memory", len(records))

    # 3. Statistical Analysis
    logger.info("Step 3/5: Computing dataset statistics & pixel metrics...")
    analyzer = FER2013Statistics()
    stats = analyzer.compute_statistics(records)
    stats_file = reports_dir / "dataset_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(stats.to_json(indent=2))
    logger.info("Saved dataset statistics to: %s", stats_file)

    # 4. Duplicate & Cross-Split Leakage Analysis
    logger.info("Step 4/5: Running exact duplicate and cross-split leakage audit...")
    dup_result = analyzer.compute_duplicates(records)
    dup_file = reports_dir / "duplicate_report.json"
    leak_file = reports_dir / "leakage_report.json"

    with open(dup_file, "w", encoding="utf-8") as f:
        f.write(dup_result.to_json(indent=2))

    leakage_summary = {
        "dataset": "FER2013",
        "total_samples": dup_result.total_samples,
        "unique_samples": dup_result.unique_samples,
        "exact_duplicates": dup_result.exact_duplicates_count,
        "cross_split_duplicates": (
            dup_result.train_val_overlap
            + dup_result.train_test_overlap
            + dup_result.val_test_overlap
        ),
        "train_val_overlap": dup_result.train_val_overlap,
        "train_test_overlap": dup_result.train_test_overlap,
        "val_test_overlap": dup_result.val_test_overlap,
        "leakage_risk_detected": dup_result.cross_split_leakage_detected,
        "leakage_mitigation_recommendation": (
            "During Phase 03 preprocessing, deduplicate cross-split samples or retain only first occurrences to eliminate evaluation data contamination."
        ),
        "details": dup_result.details,
    }
    with open(leak_file, "w", encoding="utf-8") as f:
        json.dump(leakage_summary, f, indent=2)

    logger.info("Saved duplicate report to: %s", dup_file)
    logger.info("Saved leakage report to: %s", leak_file)

    # 5. Visualizations & Sample Grids
    logger.info("Step 5/5: Generating high-resolution EDA charts and sample grids...")
    analyzer.generate_charts(stats, charts_dir)
    analyzer.generate_sample_grids(records, samples_dir, seed=seed)

    # Final Formatted Console Output
    print("\n" + "=" * 50)
    print("FER2013 DATASET ANALYSIS")
    print("=" * 50)
    print("\nDataset:")
    print("FER2013")
    print("\nTotal Samples:")
    print(f"{stats.total_samples:,}")
    print("\nClasses:")
    print(stats.num_classes)
    print("\nImage Size:")
    print(
        f"Image Size:\n{stats.image_dimensions.get('min_width')} x {stats.image_dimensions.get('min_height')}"
    )
    print("\nChannels:")
    print(stats.image_dimensions.get("channels"))
    print("\n--------------------------------------------------")
    print("CLASS DISTRIBUTION")
    print("--------------------------------------------------\n")
    for name in EMOTION_NAMES:
        c_count = stats.classes.get(name, 0)
        c_pct = stats.class_percentages.get(name, 0.0)
        print(f"{name.capitalize():<12} {c_count:>6,} ({c_pct:>5.1f}%)")

    print("\n--------------------------------------------------")
    print("SPLITS")
    print("--------------------------------------------------\n")
    for s_name, s_count in stats.splits.items():
        s_pct = stats.split_percentages.get(s_name, 0.0)
        print(f"{s_name.capitalize():<12} {s_count:>6,} ({s_pct:>5.1f}%)")

    print("\n--------------------------------------------------")
    print("DATA QUALITY")
    print("--------------------------------------------------\n")
    print(f"Invalid Records:   {val_result.invalid_records}")
    print(f"Corrupt Images:    {val_result.corrupt_images}")
    print(f"Invalid Labels:    {len(val_result.invalid_labels)}")
    print(f"Missing Records:   {val_result.missing_records}")

    print("\n--------------------------------------------------")
    print("DUPLICATES")
    print("--------------------------------------------------\n")
    print(f"Exact Duplicates:         {dup_result.exact_duplicates_count:,}")
    print(f"Duplicate Groups:         {dup_result.duplicate_groups_count:,}")
    print(f"Cross-Split Duplicates:   {leakage_summary['cross_split_duplicates']}")
    print(f"  - Train <-> Val:        {dup_result.train_val_overlap}")
    print(f"  - Train <-> Test:       {dup_result.train_test_overlap}")
    print(f"  - Val <-> Test:         {dup_result.val_test_overlap}")

    print("\n--------------------------------------------------")
    print("STATUS")
    print("--------------------------------------------------\n")
    print("Dataset validated.")
    print("EDA completed.")
    print("Ready for Phase 03.")
    print("=" * 50 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete EDA on FER2013 dataset")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/fer2013/fer2013.csv",
        help="Path to FER2013 CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/interim/fer2013/eda",
        help="Output directory for charts, samples, and reports",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )
    args = parser.parse_args()

    run_eda(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
