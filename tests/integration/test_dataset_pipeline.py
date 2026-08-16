"""Integration test for FER2013 dataset pipeline and end-to-end EDA execution."""

import tempfile
from pathlib import Path

from ml.datasets.fer2013.loader import FER2013Loader
from ml.datasets.fer2013.parser import DatasetRecord
from ml.datasets.fer2013.statistics import FER2013Statistics
from ml.datasets.fer2013.validator import FER2013Validator


def test_end_to_end_synthetic_eda_pipeline() -> None:
    """Test full Loader -> Validator -> Statistics -> Visualizer flow on synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        csv_file = tmp_path / "fer2013.csv"
        charts_dir = tmp_path / "charts"
        samples_dir = tmp_path / "samples"

        # Create synthetic CSV with 14 rows (2 per class across splits)
        rows = ["emotion,pixels,Usage"]
        for label in range(7):
            for split_idx, split_name in enumerate(["Training", "PublicTest"]):
                pixel_str = " ".join([str((label * 30 + split_idx * 10) % 256)] * 2304)
                rows.append(f"{label},{pixel_str},{split_name}")

        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("\n".join(rows))

        # 1. Validation
        validator = FER2013Validator()
        val_result = validator.validate_dataset(csv_file)
        assert val_result.is_valid is True
        assert val_result.total_records == 14
        assert val_result.valid_records == 14

        # 2. Loading
        loader = FER2013Loader(csv_file)
        records = loader.load_records()
        assert len(records) == 14

        # 3. Statistics
        stats_analyzer = FER2013Statistics()
        stats = stats_analyzer.compute_statistics(records)
        assert stats.total_samples == 14
        assert stats.num_classes == 7
        assert stats.splits["train"] == 7
        assert stats.splits["val"] == 7

        # 4. Charts & Sample Grids Generation
        stats_analyzer.generate_charts(stats, charts_dir)
        stats_analyzer.generate_sample_grids(records, samples_dir, seed=42)

        assert (charts_dir / "class_distribution.png").exists()
        assert (charts_dir / "split_distribution.png").exists()
        assert (charts_dir / "pixel_distribution.png").exists()
        assert (charts_dir / "image_dimensions.png").exists()
        assert (samples_dir / "class_samples.png").exists()
        assert (samples_dir / "random_samples.png").exists()


def test_real_dataset_loader_if_present() -> None:
    """Test loading records from the raw dataset file if present on disk."""
    raw_csv = Path("data/raw/fer2013/fer2013.csv")
    if not raw_csv.exists():
        return

    loader = FER2013Loader(raw_csv)
    # Stream first 10 records
    sample_records = loader.load_records(limit=10)
    assert len(sample_records) == 10
    for r in sample_records:
        assert isinstance(r, DatasetRecord)
        assert r.shape == (48, 48)
        assert 0 <= r.label <= 6
        assert r.split in {"train", "val", "test"}
