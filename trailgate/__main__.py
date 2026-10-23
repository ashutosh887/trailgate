"""Trailgate CLI.

  python -m trailgate run <folder or files> [--burst-gap 10 | --no-bursts] [--no-bedrock] [--no-alerts]
                         [--json out.jsonl] [--digest] [--limit N]
  python -m trailgate run --manifest data/manifests/heldout_bursts.csv [--images data/images]
  python -m trailgate agent "Process tonight's card in data/images and tell me what needs me"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import bedrock as nova


def load_bursts(args: argparse.Namespace) -> list:
    from .bursts import Burst, group_bursts, manifest_bursts
    from .pipeline import ROOT, find_frames

    if args.manifest:
        bursts = manifest_bursts(Path(args.manifest), Path(args.images or ROOT / "data" / "images"))
    else:
        bursts = group_bursts(find_frames(args.paths), args.burst_gap)
    if args.no_bursts:
        bursts = [
            Burst([p], [label], [light])
            for b in bursts
            for p, label, light in zip(b.paths, b.labels, b.lightings, strict=True)
        ]
    return bursts[: args.limit] if args.limit else bursts


def print_result(r) -> None:
    escalate = f"{r.escalate:.2f}" if r.escalate is not None else "  - "
    species = r.species or ("-" if not r.calls else "error")
    print(f"{r.tier:<8} p={r.present:.2f} e={escalate} {r.difficulty:<5} {species:<14} ${r.cost_usd:.5f}  {r.frame}")
    if r.difficulty in ("unknown", "n/a"):
        print(f"         {r.reason}")
    elif r.reason.startswith("burst:"):
        print(f"         {r.reason}")
    if r.alert:
        print(f"         alert: {r.alert}")
    for call in r.calls:
        if call.error:
            print(f"         {call.model_id}: {call.error[:110]}")


def run(args: argparse.Namespace) -> int:
    from .pipeline import Triage

    if not args.paths and not args.manifest:
        print("give image paths or --manifest", file=sys.stderr)
        return 2
    bursts = load_bursts(args)
    if not bursts:
        print("no images found", file=sys.stderr)
        return 1
    triage = Triage(bedrock=not args.no_bedrock, notify=not args.no_alerts)
    t = triage.thresholds
    n_frames = sum(len(b.paths) for b in bursts)
    print(
        f"{n_frames} frames in {len(bursts)} bursts  "
        f"thresholds: skip<{t.skip_below} small>={t.small_above} escalate>={t.escalate_above}"
    )
    out = open(args.json, "w") if args.json else None
    results = []
    try:
        for i, burst in enumerate(bursts, 1):
            if len(burst.paths) > 1:
                print(f"-- burst {i}/{len(bursts)}: {len(burst.paths)} frames {burst.key}".rstrip())
            for r in triage.run_burst(burst.paths, burst.labels, burst.lightings):
                results.append(r)
                if out:
                    out.write(json.dumps(r.to_dict()) + "\n")
                    out.flush()
                print_result(r)
    finally:
        if out:
            out.close()
    tiers = Counter(r.tier for r in results)
    spent = sum(r.cost_usd for r in results)
    print("\n" + ", ".join(f"{tier} {tiers.get(tier, 0)}" for tier in ("skip", "small", "large", "escalate")))
    print(
        f"Bedrock spend ${spent:.5f}; Nova Lite on every frame ~${nova.baseline_cost('small') * len(results):.5f}, "
        f"Nova Pro on every frame ~${nova.baseline_cost('large') * len(results):.5f} (estimate, {nova.IMAGE_TOKENS_SOURCE})"
    )
    if args.digest:
        print("\n" + nova.write_digest(results))
    return 0


def agent(args: argparse.Namespace) -> int:
    from .agent import build_agent

    trailgate_agent, _ = build_agent()
    print(trailgate_agent(args.prompt))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trailgate", description="Camera-trap triage with Strands Decider 2B and Amazon Bedrock."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="triage frames with the deterministic pipeline")
    p_run.add_argument("paths", nargs="*", help="image files or folders")
    p_run.add_argument("--manifest", help="CSV with file_name (and seq_id, frame_num, label, lighting); bursts by seq_id")
    p_run.add_argument("--images", help="image folder for --manifest (default data/images)")
    p_run.add_argument("--burst-gap", type=float, default=10.0, help="seconds between frames of one burst (default 10)")
    p_run.add_argument("--no-bursts", action="store_true", help="route every frame on its own")
    p_run.add_argument("--limit", type=int, help="process only the first N bursts")
    p_run.add_argument("--no-bedrock", action="store_true", help="Decider routing only; no Bedrock calls")
    p_run.add_argument("--no-alerts", action="store_true", help="do not write or publish ranger alerts")
    p_run.add_argument("--json", metavar="OUT", help="write one FrameResult per line to this JSONL file")
    p_run.add_argument("--digest", action="store_true", help="print the morning digest at the end")
    p_run.set_defaults(func=run)
    p_agent = sub.add_parser("agent", help="ask the Strands agent")
    p_agent.add_argument("prompt")
    p_agent.set_defaults(func=agent)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
