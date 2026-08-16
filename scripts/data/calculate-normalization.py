"""Calculate exact TRAIN-only normalization statistics for FER2013.

Ensures zero data leakage by computing pixel mean and standard deviation exclusively
from the 28,709 training samples.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.datasets.fer2013.loader import FER2013Loader  # noqa: E402
from ml.preprocessing.normalization import calculate_train_normalization_stats  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("calculate-normalization")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate TRAIN-only normalization statistics")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/fer2013/fer2013.csv",
        help="Path to raw FER2013 dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/interim/fer2013/preprocessing",
        help="Directory to save JSON normalization report",
    )
    args = parser.parse_args()

    csv_path = ROOT_DIR / args.data_path
    if not csv_path.exists():
        logger.error("Dataset not found at: %s", csv_path)
        sys.exit(1)

    logger.info("Loading training records from %s...", csv_path)
    loader = FER2013Loader(csv_path)

    logger.info("Calculating exact pixel mean and standard deviation on TRAIN split only...")
    stats = calculate_train_normalization_stats(loader, split="train")

    # Output directory
    out_dir = ROOT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "normalization_report.json"

    report_data = {
        "dataset": "FER2013",
        "split": "train",
        "total_samples": stats.total_samples,
        "scaled_0_to_1": {
            "mean": stats.mean,
            "std": stats.std,
        },
        "raw_0_to_255": {
            "mean": stats.raw_mean,
            "std": stats.raw_std,
        },
        "leakage_prevention_note": (
            "Statistics calculated exclusively from the training split (28,709 samples). "
            "Zero validation or test pixels were included in this calculation."
        ),
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\n" + "=" * 54)
    print("FER2013 TRAIN-ONLY NORMALIZATION STATISTICS")
    print("=" * 54)
    print(f"Split Analyzed:            {stats.split.upper()} ONLY")
    print(f"Sample Count:              {stats.total_samples:,}")
    print("-" * 54)
    print("NORMALIZED [0.0, 1.0] TENSOR RANGE (PyTorch Standard):")
    print(f"  Mean (mu):               {stats.mean:.6f}")
    print(f"  Standard Deviation (sigma): {stats.std:.6f}")
    print("-" * 54)
    print("RAW [0, 255] PIXEL RANGE (Reference):")
    print(f"  Mean:                    {stats.raw_mean:.2f}")
    print(f"  Standard Deviation:      {stats.raw_std:.2f}")
    print("-" * 54)
    print(f"Saved Report:              {report_file}")
    print("=" * 54 + "\n")


if __name__ == "__main__":
    main()
