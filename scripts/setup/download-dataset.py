"""Setup wrapper for downloading FER2013 dataset."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "data" / "download-dataset.py"
    runpy.run_path(str(target), run_name="__main__")
