"""Fetch every frame of the camera burst (CCT sequence) behind each frame in a manifest.

CCT labels whole sequences, so a frame labelled "deer" can be the one after the deer left.
For each manifest frame this writes all frames of its sequence (the frame itself included)
to data/manifests/<name>_bursts.csv with seq_id, frame_num, seq_num_frames and anchor.

    .venv/bin/python scripts/fetch_bursts.py heldout tune
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_cct import DATA, fetch, is_infrared, load_labels

FIELDS = [
    "id",
    "file_name",
    "label",
    "location",
    "captured",
    "lighting",
    "seq_id",
    "frame_num",
    "seq_num_frames",
    "anchor",
]


def burst_rows(labels: dict, manifest: list[dict]) -> list[dict]:
    names = {c["id"]: c["name"] for c in labels["categories"]}
    per_image = defaultdict(set)
    for a in labels["annotations"]:
        per_image[a["image_id"]].add(names[a["category_id"]])
    by_file = {i["file_name"]: i for i in labels["images"]}
    by_seq = defaultdict(list)
    for i in labels["images"]:
        by_seq[i["seq_id"]].append(i)
    rows, seen = [], set()
    for m in manifest:
        seq = by_file[m["file_name"]]["seq_id"]
        for img in sorted(by_seq[seq], key=lambda i: i["frame_num"]):
            if img["file_name"] in seen:
                continue
            seen.add(img["file_name"])
            rows.append(
                {
                    "id": img["id"],
                    "file_name": img["file_name"],
                    "label": "+".join(sorted(per_image.get(img["id"], {"unlabelled"}))),
                    "location": str(img["location"]),
                    "captured": img.get("date_captured", ""),
                    "seq_id": seq,
                    "frame_num": img["frame_num"],
                    "seq_num_frames": img["seq_num_frames"],
                    "anchor": int(
                        img["file_name"] == m["file_name"] or img["file_name"] in {x["file_name"] for x in manifest}
                    ),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="+")
    args = ap.parse_args()
    labels = load_labels()
    images = DATA / "images"
    for name in args.names:
        with open(DATA / "manifests" / f"{name}.csv", newline="") as f:
            manifest = list(csv.DictReader(f))
        rows = burst_rows(labels, manifest)
        todo = [r for r in rows if not (images / r["file_name"]).exists()]
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda r: fetch(r, images), todo))
        for r in rows:
            r["lighting"] = "night_ir" if is_infrared(images / r["file_name"]) else "day"
        with open(DATA / "manifests" / f"{name}_bursts.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(
            f"{name}: {len(manifest)} frames, {len({r['seq_id'] for r in rows})} sequences, "
            f"{len(rows)} burst frames ({len(todo)} fetched)"
        )


if __name__ == "__main__":
    main()
