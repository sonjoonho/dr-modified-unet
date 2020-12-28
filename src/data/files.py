"""Functions and utilities for file operations."""
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class MaskPaths:
    ma: List[Path]
    he: List[Path]
    ex: List[Path]
    se: List[Path]
    od: List[Path]

    def total(self):
        return sum(len(m) for m in [self.ma, self.he, self.ex, self.se, self.od])


def filepaths(path: Path) -> List[Path]:
    """Returns the paths of all files in the specified directory."""

    assert path.is_dir(), f"{path} is not a directory"

    # Sort to ensure that the images and masks match up correctly.
    return sorted([f.resolve() for f in path.glob("*") if f.is_file()])
