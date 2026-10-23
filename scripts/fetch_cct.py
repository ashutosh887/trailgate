"""Build the Trailgate evaluation sets from Caltech Camera Traps.

Splits are made by camera location, so no camera that informed question
wording or thresholds appears in the held-out set:

  tune     80 frames from 20 tuning locations, natural label mix
  heldout  600 frames from the remaining locations, natural label mix
  vehicle  40 frames labelled `car` from the held-out locations

Labels: https://lila.science/datasets/caltech-camera-traps (CDLA-Permissive-1.0).
Images are read from the public LILA mirror and resized to 1024 px on the long side.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

LABELS_URL = (
    "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_camera_traps.json.zip"
)
IMAGE_MIRRORS = (
    "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/{}",
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/{}",
)
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 2610


def load_labels() -> dict:
    cached = DATA / "caltech_images_20210113.json"
    if not cached.exists():
        raw = urllib.request.urlopen(LABELS_URL).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            z.extractall(DATA)
    return json.loads(cached.read_text())


def single_label_frames(labels: dict) -> list[dict]:
    names = {c["id"]: c["name"] for c in labels["categories"]}
    per_image = defaultdict(set)
    for a in labels["annotations"]:
        per_image[a["image_id"]].add(names[a["category_id"]])
    frames = []
    for img in labels["images"]:
        cats = per_image.get(img["id"], set())
        if len(cats) != 1:
            continue
        frames.append(
            {
                "id": img["id"],
                "file_name": img["file_name"],
                "label": next(iter(cats)),
                "location": str(img["location"]),
                "captured": img.get("date_captured", ""),
            }
        )
    return frames


def split(frames: list[dict], n_tune_locations: int, n_tune: int, n_heldout: int, n_vehicle: int):
    rng = random.Random(SEED)
    locations = sorted({f["location"] for f in frames})
    rng.shuffle(locations)
    tune_locs = set(locations[:n_tune_locations])
    tune_pool = [f for f in frames if f["location"] in tune_locs]
    held_pool = [f for f in frames if f["location"] not in tune_locs]
    tune = rng.sample(tune_pool, n_tune)
    heldout = rng.sample(held_pool, n_heldout)
    taken = {f["id"] for f in heldout}
    cars = [f for f in held_pool if f["label"] == "car" and f["id"] not in taken]
    vehicle = rng.sample(cars, n_vehicle)
    return {"tune": tune, "heldout": heldout, "vehicle": vehicle}


def fetch(frame: dict, out: Path) -> None:
    path = out / frame["file_name"]
    if path.exists():
        return
    raw = None
    for attempt in range(4):
        try:
            raw = urllib.request.urlopen(IMAGE_MIRRORS[attempt % 2].format(frame["file_name"]), timeout=60).read()
            break
        except OSError:
            if attempt == 3:
                raise
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((1024, 1024))
    img.save(path, "JPEG", quality=90)


def is_infrared(path: Path) -> bool:
    img = Image.open(path).convert("RGB").resize((64, 48))
    spread = [max(p) - min(p) for p in img.get_flattened_data()]
    return sum(spread) / len(spread) < 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tune", type=int, default=80)
    parser.add_argument("--heldout", type=int, default=600)
    parser.add_argument("--vehicle", type=int, default=40)
    parser.add_argument("--tune-locations", type=int, default=20)
    args = parser.parse_args()

    sets = split(single_label_frames(load_labels()), args.tune_locations, args.tune, args.heldout, args.vehicle)
    images = DATA / "images"
    images.mkdir(parents=True, exist_ok=True)
    every = {f["file_name"]: f for rows in sets.values() for f in rows}
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda f: fetch(f, images), every.values()))

    manifests = DATA / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for name, rows in sets.items():
        with open(manifests / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "file_name", "label", "location", "captured", "lighting"])
            w.writeheader()
            for f in rows:
                lighting = "night_ir" if is_infrared(images / f["file_name"]) else "day"
                w.writerow({**f, "lighting": lighting})
        empty = sum(f["label"] == "empty" for f in rows)
        print(f"{name}: {len(rows)} frames, {empty} empty, {len({f['location'] for f in rows})} locations")


if __name__ == "__main__":
    main()
