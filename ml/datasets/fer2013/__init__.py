"""FER2013 Dataset Package for Facial Expression Recognition.

Provides dataset loaders, parsers, validators, and exploratory statistics tools.
"""

from ml.datasets.fer2013.loader import FER2013Loader
from ml.datasets.fer2013.parser import (
    EMOTION_LABELS,
    EMOTION_NAMES,
    SPLIT_MAPPING,
    DatasetRecord,
    parse_csv_row,
    parse_pixels_string,
    pixels_to_hash,
)
from ml.datasets.fer2013.statistics import (
    DatasetStatisticsResult,
    DuplicateAnalysisResult,
    FER2013Statistics,
)
from ml.datasets.fer2013.validator import (
    DatasetValidationResult,
    FER2013Validator,
)

__all__ = [
    "EMOTION_LABELS",
    "EMOTION_NAMES",
    "SPLIT_MAPPING",
    "DatasetRecord",
    "DatasetStatisticsResult",
    "DatasetValidationResult",
    "DuplicateAnalysisResult",
    "FER2013Loader",
    "FER2013Statistics",
    "FER2013Validator",
    "parse_csv_row",
    "parse_pixels_string",
    "pixels_to_hash",
]
