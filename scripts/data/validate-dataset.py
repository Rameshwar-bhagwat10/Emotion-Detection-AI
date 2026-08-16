"""Validate FER2013 dataset structure, records, labels, and integrity.

CLI script that orchestrates `FER2013Validator` and outputs structured reports.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add root directory to pythonpath
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.datasets.fer2013.validator import FER2013Validator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate-dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FER2013 dataset integrity")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/fer2013/fer2013.csv",
        help="Path to FER2013 CSV file or raw dataset folder",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="data/interim/fer2013/eda/reports/validation_report.json",
        help="Path to save output validation report JSON",
    )
    args = parser.parse_args()

    data_path = ROOT_DIR / args.data_path
    output_path = ROOT_DIR / args.output_report

    logger.info("Starting validation for dataset at: %s", data_path)
    validator = FER2013Validator()
    result = validator.validate_dataset(data_path)

    # Ensure parent output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.to_json(indent=2))

    logger.info("Validation report saved to: %s", output_path)

    print("\n" + "=" * 50)
    print("FER2013 DATASET VALIDATION REPORT")
    print("=" * 50)
    print(f"Dataset Path:       {result.dataset_path}")
    print(f"Total Records:      {result.total_records:,}")
    print(f"Valid Records:      {result.valid_records:,}")
    print(f"Invalid Records:    {result.invalid_records}")
    print(f"Corrupt Images:     {result.corrupt_images}")
    print(f"Invalid Dimensions: {result.invalid_dimensions}")
    print(f"Splits Found:       {result.split_counts}")
    print(f"Class Counts:       {result.class_counts}")
    print(f"Status:             {'PASS (Valid)' if result.is_valid else 'FAIL (Invalid)'}")
    print("=" * 50 + "\n")

    if not result.is_valid:
        logger.error("Dataset validation failed with %d errors.", len(result.errors))
        for err in result.errors[:10]:
            logger.error("  - %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
