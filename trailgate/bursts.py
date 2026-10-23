"""Group frames into camera bursts: by capture time, or by the dataset's sequence id."""

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

    def __post_init__(self) -> None:
        self.labels = self.labels or [None] * len(self.paths)
        self.lightings = self.lightings or [None] * len(self.paths)


def capture_time(path: Path) -> float:
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            stamp = exif.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or exif.get(DATETIME)
        if stamp:
            return datetime.strptime(str(stamp).strip("\x00 "), EXIF_FORMAT).timestamp()
    except (OSError, ValueError):
        pass
    return path.stat().st_mtime if path.exists() else 0.0


def group_bursts(paths: list[Path], gap_s: float = 10.0, max_size: int = 10) -> list[Burst]:
    by_folder: dict[Path, list[tuple[float, Path]]] = {}
    for p in paths:
        by_folder.setdefault(p.parent, []).append((capture_time(p), p))
    bursts: list[Burst] = []
    for folder in sorted(by_folder):
        current: list[Path] = []
        last = None
        for stamp, p in sorted(by_folder[folder]):
            if current and (stamp - last > gap_s or len(current) >= max_size):
                bursts.append(Burst(current))
                current = []
            current.append(p)
            last = stamp
        if current:
            bursts.append(Burst(current))
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
        bursts.append(Burst(
            [images / r["file_name"] for r in members],
            [r.get("label") or None for r in members],
            [r.get("lighting") or None for r in members],
            key,
        ))
    return bursts
