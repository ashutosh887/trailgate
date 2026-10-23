"""Bedrock baselines: Nova Lite and Nova Pro species labels on every held-out frame.

Results are cached to results/nova_{lite,pro}.jsonl (one row per frame, keyed by file_name)
so the run resumes; rows with an error are retried on the next run. scripts/evaluate.py merges
them in when present.

    .venv/bin/python scripts/run_nova_baseline.py                  # both tiers, all held-out frames
    .venv/bin/python scripts/run_nova_baseline.py --tier small --limit 1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patterns.presence_router.tune_questions import read_manifest
from trailgate.bedrock import label_species

RESULTS = ROOT / "results"
FILES = {"small": "nova_lite.jsonl", "large": "nova_pro.jsonl"}


def load_done(path: Path) -> dict:
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if not row.get("error"):
                    done[row["file_name"]] = row
    return done


def run_tier(tier: str, rows: list[dict], limit: int | None, max_errors: int) -> dict:
    path = RESULTS / FILES[tier]
    done = load_done(path)
    todo = [r for r in rows if r["file_name"] not in done][:limit]
    errors = 0
    with open(path, "a") as out:
        for i, r in enumerate(todo, 1):
            species, confidence, call = label_species((ROOT / "data/images" / r["file_name"]).read_bytes(), tier)
            row = {
                "file_name": r["file_name"],
                "label": r["label"],
                "lighting": r["lighting"],
                "species": species,
                "confidence": confidence,
                **asdict(call),
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            if call.error:
                errors += 1
                print(f"{tier} {r['file_name']}: {call.error}", file=sys.stderr)
                if errors >= max_errors:
                    print(f"{tier}: stopping after {errors} errors", file=sys.stderr)
                    break
            elif i % 25 == 0:
                print(f"{tier}: {i}/{len(todo)}", file=sys.stderr)
    final = load_done(path)
    return {
        "tier": tier,
        "file": str(path.relative_to(ROOT)),
        "ok": len(final),
        "of": len(rows),
        "errors_this_run": errors,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", choices=["small", "large", "both"], default="both")
    ap.add_argument("--manifest", default=str(ROOT / "data/heldout.csv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-errors", type=int, default=3)
    args = ap.parse_args(argv)
    rows = read_manifest(Path(args.manifest))
    tiers = ["small", "large"] if args.tier == "both" else [args.tier]
    for tier in tiers:
        print(json.dumps(run_tier(tier, rows, args.limit, args.max_errors)))


if __name__ == "__main__":
    main()
