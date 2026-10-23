"""Trailgate CLI.

python -m trailgate run <folder or files> [--burst-gap 10 | --no-bursts] [--no-bedrock] [--no-alerts]
                       [--json out.jsonl] [--digest] [--limit N]
python -m trailgate run --manifest data/heldout_bursts.csv [--images data/images]
python -m trailgate agent "Process tonight's card in data/images and tell me what needs me"
"""

from __future__ import annotations

import argparse
import json
import os
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
            Burst([p], [label], [light], dated=b.dated)
            for b in bursts
            for p, label, light in zip(b.paths, b.labels, b.lightings, strict=True)
        ]
    return bursts[: args.limit] if args.limit else bursts


def is_unavailable(call) -> bool:
    return bool(call.error) and call.error.startswith(nova.UNAVAILABLE)


def species_cell(r) -> str:
    errors = [c for c in r.calls if c.error]
    if r.species or not errors:
        return r.species or "-"
    return nova.UNAVAILABLE if all(map(is_unavailable, errors)) else "error"


def print_result(r) -> None:
    if r.decider_error:
        scores = "Decider error    "
    else:
        escalate = f"{r.escalate:.2f}" if r.escalate is not None else "  - "
        scores = f"p={r.present:.2f} e={escalate}"
    spend = f"${r.cost_usd:.5f}" + ("" if all(c.priced for c in r.calls) else " unpriced")
    print(f"{r.tier:<8} {scores} {r.difficulty:<5} {species_cell(r):<14} {spend}  {r.frame}")
    if r.difficulty in ("unknown", "n/a") or r.reason.startswith("burst:"):
        print(f"         {r.reason}")
    if r.alert:
        print(f"         alert: {r.alert}")
    for call in r.calls:
        if call.error and not is_unavailable(call):
            print(f"         {call.model_id}: {call.error[:110]}")


def print_summary(results: list) -> None:
    tiers = Counter(r.tier for r in results)
    calls = [c for r in results for c in r.calls]
    spent = sum(c.cost_usd for c in calls)
    unpriced = sum(not c.priced for c in calls)
    print("\n" + ", ".join(f"{tier} {tiers.get(tier, 0)}" for tier in ("skip", "small", "large", "escalate")))
    failed = sum(1 for r in results if r.decider_error)
    if failed:
        print(f"Decider error on {failed} frames; sent to the large model")
    for model, n in Counter(c.model_id for c in calls if is_unavailable(c)).items():
        print(f"Bedrock unavailable for {model}: {n} calls skipped")
    print(
        f"Bedrock spend ${spent:.5f}" + (f" ({unpriced} calls unpriced)" if unpriced else "") + "; "
        f"Nova Lite on every frame ~${nova.baseline_cost('small') * len(results):.5f}, "
        f"Nova Pro on every frame ~${nova.baseline_cost('large') * len(results):.5f} (estimate, {nova.IMAGE_TOKENS_SOURCE})"
    )


def run(args: argparse.Namespace) -> int:
    from .pipeline import Triage, missing_frames

    if not args.paths and not args.manifest:
        print("give image paths or --manifest", file=sys.stderr)
        return 2
    bursts = load_bursts(args)
    if not bursts:
        print("no images found", file=sys.stderr)
        return 1
    missing = missing_frames([p for b in bursts for p in b.paths])
    if missing:
        shown = "\n  ".join(map(str, missing[:5])) + ("\n  ..." if len(missing) > 5 else "")
        print(f"{len(missing)} frames not found:\n  {shown}", file=sys.stderr)
        return 1
    undated = sum(len(b.paths) for b in bursts if not b.dated)
    if undated:
        print(f"warning: {undated} frames have no EXIF capture time; each is routed as its own burst", file=sys.stderr)
    if args.no_bedrock and not args.no_alerts and os.environ.get("TRAILGATE_ALERT_TOPIC_ARN"):
        print("--no-bedrock: alerts go to the local outbox only; SNS is not published", file=sys.stderr)
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
    print_summary(results)
    if args.digest:
        print("\n" + nova.write_digest(results))
    return 0


def agent(args: argparse.Namespace) -> int:
    from botocore.exceptions import BotoCoreError, ClientError
    from strands.types.exceptions import ModelThrottledException

    from .agent import agent_model_id, build_agent

    trailgate_agent, _ = build_agent()
    try:
        print(trailgate_agent(args.prompt))
    except (ClientError, BotoCoreError, ModelThrottledException) as exc:
        detail = nova.error_text(exc)
        print(
            f"The agent's model {agent_model_id()} is not available on Bedrock ({detail[:200]}).\n"
            "The deterministic pipeline needs no agent model: python -m trailgate run <folder> (add --no-bedrock to skip Bedrock)",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trailgate", description="Camera-trap triage with Strands Decider 2B and Amazon Bedrock."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="triage frames with the deterministic pipeline")
    p_run.add_argument("paths", nargs="*", help="image files or folders")
    p_run.add_argument(
        "--manifest", help="CSV with file_name (and seq_id, frame_num, label, lighting); bursts by seq_id"
    )
    p_run.add_argument("--images", help="image folder for --manifest (default data/images)")
    p_run.add_argument("--burst-gap", type=float, default=10.0, help="seconds between frames of one burst (default 10)")
    p_run.add_argument("--no-bursts", action="store_true", help="route every frame on its own")
    p_run.add_argument("--limit", type=int, help="process only the first N bursts")
    p_run.add_argument(
        "--no-bedrock", action="store_true", help="Decider routing only; no Bedrock calls and no SNS (outbox only)"
    )
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
