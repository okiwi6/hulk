from pathlib import Path
from typing import Generator

from dvc.api import DVCFileSystem

# dvc list . data/HULKs/events/2021-05-GermanOpenReplacementEvent2021/ -R --dvc-only | rg ".png"


class DvcImageQuery:
    def __init__(self, dvc_root: Path):
        fs = DVCFileSystem(dvc_root)
        paths = fs.find("../data/HULKs/events/", detail=False, dvc_only=True)
        png_files = [p for p in paths if p.endswith(".png")]

        print(f"Found {len(png_files)} PNG files in {dvc_root}")
