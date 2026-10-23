"""Group frames into camera bursts: by EXIF capture time, or by the dataset's sequence id.

A frame without an EXIF capture time is its own burst (`Burst.dated` is False); file
modification times are never used, since copying a card rewrites them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image

EXIF_IFD = 0x8769
DATETIME_ORIGINAL = 0x9003
DATETIME = 0x0132
EXIF_FORMAT = "%Y:%m:%d %H:%M:%S"


@dataclass
class Burst:
    paths: list[Path]
    labels: list[str | None] = field(default_factory=list)
    lightings: list[str | None] = field(default_factory=list)
    key: str = ""
    dated: bool = True

    def __post_init__(self) -> None:
        self.labels = self.labels or [None] * len(self.paths)
        self.lightings = self.lightings or [None] * len(self.paths)


def capture_time(path: Path) -> float | None:
    """EXIF DateTimeOriginal (or DateTime) as a timestamp, or None if the frame has neither."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            stamp = exif.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or exif.get(DATETIME)
        if stamp:
            return datetime.strptime(str(stamp).strip("\x00 "), EXIF_FORMAT).timestamp()
    except (OSError, ValueError):
        pass
    return None


def group_bursts(paths: list[Path], gap_s: float = 10.0, max_size: int = 10) -> list[Burst]:
    by_folder: dict[Path, list[tuple[float | None, Path]]] = {}
    for p in paths:
        by_folder.setdefault(p.parent, []).append((capture_time(p), p))
    bursts: list[Burst] = []
    for folder in sorted(by_folder):
        dated = sorted((stamp, p) for stamp, p in by_folder[folder] if stamp is not None)
        current: list[Path] = []
        last = 0.0
        for stamp, p in dated:
            if current and (stamp - last > gap_s or len(current) >= max_size):
                bursts.append(Burst(current))
                current = []
            current.append(p)
            last = stamp
        if current:
            bursts.append(Burst(current))
        bursts.extend(
            Burst([p], dated=False) for stamp, p in sorted(by_folder[folder], key=lambda x: x[1]) if stamp is None
        )
    return bursts


def manifest_bursts(manifest: Path, images: Path) -> list[Burst]:
    with open(manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get("seq_id") or row["file_name"], []).append(row)
    bursts = []
    for key, members in groups.items():
        members.sort(key=lambda r: int(r.get("frame_num") or 0))
        bursts.append(
            Burst(
                [images / r["file_name"] for r in members],
                [r.get("label") or None for r in members],
                [r.get("lighting") or None for r in members],
                key,
            )
        )
    return bursts
