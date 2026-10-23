"""Test one multi-image Decider call per burst against max P(animal) over per-frame calls.

Tuning bursts only (2-4 frames; longer bursts use their first 4, Decider's image limit).
The presence question is asked once with every burst frame attached. Results go to
results/burst_multi_image.{jsonl,json}; scripts/evaluate.py reports them.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patterns.presence_router.questions import presence_question
from patterns.presence_router.router import DeciderClient
from patterns.presence_router.tune_questions import auc, read_manifest

RESULTS = ROOT / "results"
IMAGES = ROOT / "data/images"
STATE = "Frames from one burst of a wildlife camera trap, taken seconds apart."
MAX_IMAGES = 4


def question() -> dict:
    q = presence_question()
    q["instructions"] = "Is there an animal anywhere in these photos, even partly hidden or at the edge?"
    return {"present": q}


def main() -> dict:
    thresholds = json.loads((RESULTS / "thresholds.json").read_text())
    skip_below = thresholds["skip_below"]
    with open(RESULTS / "label_audit.csv", newline="") as f:
        audit = {a["file_name"]: a["audited_label"] for a in csv.DictReader(f)}
    per_frame = {}
    for line in (RESULTS / "tune_answers_production.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            per_frame[row["file_name"]] = row
    bursts = defaultdict(list)
    label = {}
    for b in read_manifest(ROOT / "data/manifests/tune_bursts.csv"):
        bursts[b["seq_id"]].append(b)
        if b["anchor"] == "1":
            label[b["seq_id"]] = audit.get(b["file_name"], b["label"])
    chosen = {
        s: sorted(g, key=lambda b: int(b["frame_num"]))[:MAX_IMAGES]
        for s, g in bursts.items()
        if len(g) >= 2 and label[s] not in ("car", "person")
    }

    cache_path = RESULTS / "burst_multi_image.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["seq_id"]] = row
    client = DeciderClient()
    q = question()
    with open(cache_path, "a") as out:
        for seq, frames in chosen.items():
            if seq in cache:
                continue
            images = [(IMAGES / f["file_name"]).read_bytes() for f in frames]
            started = time.perf_counter()
            response = client.ask(q, images, STATE)
            row = {
                "seq_id": seq,
                "files": [f["file_name"] for f in frames],
                "label": label[seq],
                "multi": response["answers"]["present"]["noul"],
                "wall_latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            cache[seq] = row

    rows = [cache[s] for s in chosen]
    for r in rows:
        r["max"] = max(per_frame[f]["answers"]["present"]["noul"] for f in r["files"])
        r["frames_latency_ms"] = sum(per_frame[f]["wall_latency_ms"] for f in r["files"])
        r["animal"] = r["label"] != "empty"
    pos = [r for r in rows if r["animal"]]
    neg = [r for r in rows if not r["animal"]]
    lat_multi = sorted(r["wall_latency_ms"] for r in rows)
    lat_frames = sorted(r["frames_latency_ms"] for r in rows)
    result = {
        "n_bursts": len(rows),
        "n_animal_bursts": len(pos),
        "n_empty_bursts": len(neg),
        "auc_multi": round(auc([r["multi"] for r in pos], [r["multi"] for r in neg]), 4),
        "auc_max": round(auc([r["max"] for r in pos], [r["max"] for r in neg]), 4),
        "mean_abs_diff": round(sum(abs(r["multi"] - r["max"]) for r in rows) / len(rows), 4),
        "agree_at_skip": sum((r["multi"] < skip_below) == (r["max"] < skip_below) for r in rows),
        "animal_below_multi": sum(r["multi"] < skip_below for r in pos),
        "animal_below_max": sum(r["max"] < skip_below for r in pos),
        "empty_below_multi": sum(r["multi"] < skip_below for r in neg),
        "empty_below_max": sum(r["max"] < skip_below for r in neg),
        "latency_multi_p50": lat_multi[len(lat_multi) // 2],
        "latency_frames_p50": lat_frames[len(lat_frames) // 2],
        "skip_below": skip_below,
        "state": STATE,
        "question": q,
        "rows": [
            {k: r[k] for k in ("seq_id", "label", "multi", "max", "wall_latency_ms", "frames_latency_ms")} for r in rows
        ],
    }
    as_good = result["auc_multi"] >= result["auc_max"] and result["animal_below_multi"] <= result["animal_below_max"]
    result["adopted"] = "yes" if as_good else "no: per-frame calls with max over the burst are kept"
    (RESULTS / "burst_multi_image.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "question")}, indent=2))
    return result


if __name__ == "__main__":
    main()
