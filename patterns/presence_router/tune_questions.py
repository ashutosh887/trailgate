"""Score several wordings of the presence and escalation questions on a labelled set.

Mike Chambers' method: the wording (and explicit true/false criteria) is the main
lever, so measure it. Every wording is asked in ONE Decider call per frame, so the
image is encoded once. Answers are cached to a JSONL file and the run resumes.

    python -m patterns.presence_router.tune_questions \
        --manifest data/tune.csv --negative empty --ignore car,person \
        --escalate-positive car,person --freeze results/thresholds.json

A frame counts as a positive for presence when its label is in --positive, or, when
--positive is empty, when its label is not in --negative. Labels in --ignore are left out
of the presence scores. --audit applies hand-checked relabels (file_name, audited_label).
--freeze writes the chosen wording and thresholds to results/thresholds.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

from .questions import difficulty_question
from .router import DeciderClient

ROOT = Path(__file__).resolve().parents[2]
STATE = "A single frame from a wildlife camera trap."

PRESENCE_WORDINGS = {
    "p_criteria": {
        "type": "noul",
        "instructions": "Is there an animal anywhere in this photo, even partly hidden or at the edge?",
        "criteria": {
            "true": "an animal or part of one is visible",
            "false": "only vegetation, ground, sky, or camera artefacts",
        },
    },
    "p_plain": {
        "type": "noul",
        "instructions": "Is there an animal anywhere in this photo, even partly hidden or at the edge?",
    },
    "p_short": {"type": "noul", "instructions": "Is there an animal in this photo?"},
    "p_criteria_hard": {
        "type": "noul",
        "instructions": "Is there an animal or bird anywhere in this photo, even small, far away, dark or at the edge?",
        "criteria": {
            "true": "an animal or bird, or any part of one (eyes, legs, tail, body), is visible, including blurred, distant or infrared-lit ones",
            "false": "only vegetation, ground, rocks, sky, shadows, or camera artefacts such as glare, rain, dust or motion blur",
        },
    },
    "p_empty_inverted": {
        "type": "noul",
        "instructions": "Is this photo empty, showing only the background scene?",
        "criteria": {
            "true": "only vegetation, ground, sky, or camera artefacts; no animal, person or vehicle",
            "false": "an animal, person or vehicle, or part of one, is visible",
        },
    },
}

ESCALATE_WORDINGS = {
    "e_plain": {"type": "noul", "instructions": "Is there a person or a vehicle in this photo?"},
    "e_criteria": {
        "type": "noul",
        "instructions": "Is there a person or a vehicle in this photo?",
        "criteria": {
            "true": "a person, car, truck, motorbike or other vehicle, or part of one, is visible",
            "false": "only animals, vegetation, ground, sky, or camera artefacts",
        },
    },
    "e_human_activity": {
        "type": "noul",
        "instructions": "Is there any sign of people in this photo: a person, a vehicle, or headlights?",
        "criteria": {
            "true": "a person, a vehicle, or vehicle headlights are visible",
            "false": "only animals and the natural scene",
        },
    },
}

INVERTED = {"p_empty_inverted"}


def all_questions() -> dict:
    return {**PRESENCE_WORDINGS, **ESCALATE_WORDINGS, "difficulty": difficulty_question()}


def question_signature(questions: dict, state: str) -> str:
    return hashlib.sha1(json.dumps([questions, state], sort_keys=True).encode()).hexdigest()[:12]


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_cache(path: Path, signature: str) -> dict:
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("signature") == signature:
                cache[row["file_name"]] = row
    return cache


def ask_all(
    rows: list[dict],
    questions: dict,
    cache_path: Path,
    images: Path,
    state: str = STATE,
    client: DeciderClient | None = None,
) -> dict:
    """Ask `questions` once per frame, caching raw answers; returns {file_name: cached row}."""
    client = client or DeciderClient()
    signature = question_signature(questions, state)
    cache = load_cache(cache_path, signature)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if r["file_name"] not in cache]
    with open(cache_path, "a") as out:
        for i, r in enumerate(todo, 1):
            data = (images / r["file_name"]).read_bytes()
            for attempt in range(3):
                try:
                    started = time.perf_counter()
                    response = client.ask(questions, [data], state)
                    wall_ms = (time.perf_counter() - started) * 1000
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise
                    print(f"retry {r['file_name']}: {exc}", file=sys.stderr)
                    time.sleep(5)
            row = {
                "file_name": r["file_name"],
                "signature": signature,
                "answers": response["answers"],
                "usage": response.get("usage"),
                "server_latency_ms": response.get("latency_ms"),
                "wall_latency_ms": round(wall_ms, 1),
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            cache[r["file_name"]] = row
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} frames asked", file=sys.stderr)
    return cache


def score_of(answers: dict, key: str) -> float:
    value = answers[key]["noul"]
    return 1.0 - value if key in INVERTED else value


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_accuracy(pos: list[float], neg: list[float]) -> tuple[float, float]:
    candidates = sorted(set(pos + neg + [0.0, 1.01]))
    best = (0.0, 0.5)
    total = len(pos) + len(neg)
    for t in candidates:
        acc = (sum(p >= t for p in pos) + sum(n < t for n in neg)) / total
        if acc > best[0]:
            best = (acc, t)
    return best


def zero_miss_threshold(pos: list[float], miss_budget: int = 0, step: float = 0.01) -> float:
    """Highest threshold t on a `step` grid with at most `miss_budget` positives scoring below t."""
    ordered = sorted(pos)
    limit = ordered[miss_budget] if miss_budget < len(ordered) else 1.0
    return int(limit / step + 1e-9) * step


def wording_row(name: str, question: dict, pos: list[float], neg: list[float], ref_skip: float) -> dict:
    acc, thr = best_accuracy(pos, neg)
    t0 = zero_miss_threshold(pos, 0)
    t1 = zero_miss_threshold(pos, 1)
    return {
        "wording": name,
        "instructions": question["instructions"],
        "criteria": bool(question.get("criteria")),
        "inverted": name in INVERTED,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "auc": round(auc(pos, neg), 4) if auc(pos, neg) is not None else None,
        "mean_pos": round(sum(pos) / len(pos), 3) if pos else None,
        "mean_neg": round(sum(neg) / len(neg), 3) if neg else None,
        "best_accuracy": round(acc, 4),
        "best_threshold": round(thr, 4),
        "ref_threshold": ref_skip,
        "recall_at_ref": round(sum(p >= ref_skip for p in pos) / len(pos), 4) if pos else None,
        "neg_below_ref": round(sum(n < ref_skip for n in neg) / len(neg), 4) if neg else None,
        "zero_miss_threshold": round(t0, 2),
        "neg_below_zero_miss": round(sum(n < t0 for n in neg) / len(neg), 4) if neg else None,
        "one_miss_threshold": round(t1, 2),
        "neg_below_one_miss": round(sum(n < t1 for n in neg) / len(neg), 4) if neg else None,
    }


def is_positive(label: str, positive: set[str], negative: set[str]) -> bool:
    return label in positive if positive else label not in negative


def split_scores(rows, cache, key, label_col, positive, negative, ignore=frozenset()):
    pos, neg = [], []
    for r in rows:
        if r[label_col] in ignore:
            continue
        s = score_of(cache[r["file_name"]]["answers"], key)
        (pos if is_positive(r[label_col], positive, negative) else neg).append(s)
    return pos, neg


def markdown_table(title: str, rows: list[dict], pos_name: str, neg_name: str) -> str:
    lines = [
        f"### {title}",
        "",
        f"Positives: {pos_name} (n={rows[0]['n_pos']}). Negatives: {neg_name} (n={rows[0]['n_neg']}).",
        "",
        "| Wording | Criteria | AUC | Mean score pos / neg | Best accuracy (at t) | Recall at t="
        + f"{rows[0]['ref_threshold']}"
        + " | Neg below t="
        + f"{rows[0]['ref_threshold']}"
        + " | Zero-miss t | Neg below zero-miss t |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        inst = r["instructions"] + (" (score = 1 - answer)" if r["inverted"] else "")
        lines.append(
            f"| `{r['wording']}`: {inst} | {'yes' if r['criteria'] else 'no'} | {r['auc']:.3f} | "
            f"{r['mean_pos']:.2f} / {r['mean_neg']:.2f} | {r['best_accuracy']:.1%} ({r['best_threshold']:.2f}) | "
            f"{r['recall_at_ref']:.1%} | {r['neg_below_ref']:.1%} | {r['zero_miss_threshold']:.2f} | {r['neg_below_zero_miss']:.1%} |"
        )
    return "\n".join(lines)


def apply_audit(rows: list[dict], path: Path, label_col: str) -> int:
    if not path.exists():
        return 0
    with open(path, newline="") as f:
        audit = {a["file_name"]: a["audited_label"] for a in csv.DictReader(f)}
    changed = 0
    for r in rows:
        new = audit.get(r["file_name"])
        if new and new != r[label_col]:
            r[label_col] = new
            changed += 1
    return changed


def freeze(rows, cache, label_col, positive, negative, ignore, esc_positive, presence, escalation, step=0.01) -> dict:
    """Pick wordings by AUC (ties: more negatives skipped at zero misses) and set thresholds by fixed rules."""
    p_best = max(presence, key=lambda r: (r["auc"], r["neg_below_zero_miss"]))
    e_best = max(escalation, key=lambda r: (r["auc"], (r["min_pos"] or 0) - r["max_neg"]))
    pk, ek = p_best["wording"], e_best["wording"]
    pos, neg = split_scores(rows, cache, pk, label_col, positive, negative, ignore)
    skip_below = zero_miss_threshold(pos, 0, step)
    clear = [
        (score_of(cache[r["file_name"]]["answers"], pk), is_positive(r[label_col], positive, negative))
        for r in rows
        if r[label_col] not in ignore and cache[r["file_name"]]["answers"]["difficulty"]["choice"] == "clear"
    ]
    clear_neg = max((s for s, p in clear if not p), default=0.0)
    clear_pos = min((s for s, p in clear if p and s > clear_neg), default=1.0)
    small_above = max(skip_below, round((clear_neg + clear_pos) / 2, 2))
    epos, eneg = split_scores(rows, cache, ek, label_col, esc_positive, set())
    escalate_above = round((max(eneg) + min(epos)) / 2, 2)
    return {
        "skip_below": round(skip_below, 2),
        "small_above": small_above,
        "escalate_above": escalate_above,
        "chosen_wording": pk,
        "chosen_escalate_wording": ek,
        "method": (
            f"Tuning set only. Wording: highest AUC (escalation ties broken by the widest gap between classes). skip_below: highest {step} grid value with zero tuning animals scoring below it "
            f"(miss budget 0; lowest animal score {min(pos):.3f}). small_above: midpoint of the gap between the highest-scoring "
            f"empty frame Decider called clear ({clear_neg:.3f}) and the lowest-scoring clear animal frame above it ({clear_pos:.3f}). "
            f"escalate_above: midpoint of the gap between the highest non-person/vehicle frame ({max(eneg):.3f}) and the lowest "
            f"person/vehicle frame ({min(epos):.3f}), labels audited by eye where CCT marks people as empty."
        ),
        "n_tune": len(rows),
        "n_tune_animal": len(pos),
        "n_tune_empty": len(neg),
        "n_tune_person_or_vehicle": len(epos),
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(ROOT / "data/tune.csv"))
    ap.add_argument("--images", default=str(ROOT / "data/images"))
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive", default="", help="comma-separated presence-positive labels")
    ap.add_argument("--negative", default="empty", help="used when --positive is empty")
    ap.add_argument("--ignore", default="car,person", help="labels left out of presence scoring")
    ap.add_argument("--escalate-positive", default="car,person,human,vehicle,truck")
    ap.add_argument("--audit", default=str(ROOT / "results/label_audit.csv"))
    ap.add_argument("--freeze", default="", help="write thresholds JSON here")
    ap.add_argument("--ref-skip", type=float, default=0.35, help="reference skip threshold for recall")
    ap.add_argument("--ref-escalate", type=float, default=0.5)
    ap.add_argument("--cache", default=str(ROOT / "results/tune_answers.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results/wording"))
    args = ap.parse_args(argv)

    rows = read_manifest(Path(args.manifest))
    audited = apply_audit(rows, Path(args.audit), args.label_col) if args.audit else 0
    positive = {s for s in args.positive.split(",") if s}
    negative = {s for s in args.negative.split(",") if s}
    ignore = {s for s in args.ignore.split(",") if s}
    esc_positive = {s for s in args.escalate_positive.split(",") if s}
    questions = all_questions()
    print(f"{len(rows)} frames, {len(questions)} questions per call", file=sys.stderr)
    cache = ask_all(rows, questions, Path(args.cache), Path(args.images))

    presence = []
    for name, q in PRESENCE_WORDINGS.items():
        pos, neg = split_scores(rows, cache, name, args.label_col, positive, negative, ignore)
        presence.append(wording_row(name, q, pos, neg, args.ref_skip))
    escalation = []
    for name, q in ESCALATE_WORDINGS.items():
        pos, neg = split_scores(rows, cache, name, args.label_col, esc_positive, set())
        escalation.append(wording_row(name, q, pos, neg, args.ref_escalate))
        r = escalation[-1]
        r["max_neg"] = round(max(neg), 4)
        r["min_pos"] = round(min(pos), 4) if pos else None
        r["false_alerts_at_ref"] = sum(n >= args.ref_escalate for n in neg)
    latencies = sorted(cache[r["file_name"]]["wall_latency_ms"] for r in rows)

    pos_name = ", ".join(sorted(positive)) if positive else f"any label except {', '.join(sorted(negative))}"
    neg_name = f"labels {', '.join(sorted(negative))}" if not positive else "all other labels"
    if ignore:
        neg_name += f" (frames labelled {', '.join(sorted(ignore))} left out)"
    result = {
        "manifest": str(Path(args.manifest).relative_to(ROOT)) if Path(args.manifest).is_absolute() else args.manifest,
        "n_frames": len(rows),
        "labels_changed_by_audit": audited,
        "questions_per_call": len(questions),
        "state": STATE,
        "signature": question_signature(questions, STATE),
        "decider_wall_latency_ms_p50": latencies[len(latencies) // 2],
        "presence": presence,
        "escalation": escalation,
        "questions": questions,
    }
    md = [
        f"## Question wording on the tuning set ({len(rows)} frames, {len(questions)} questions in one Decider call per frame)",
        "",
        "Scored on the tuning locations only. AUC is the chance a random positive frame scores above a random negative one. "
        '"Zero-miss t" is the highest skip threshold (0.01 grid) at which no positive frame in this set falls below it.',
        "",
        markdown_table("Presence: is there an animal?", presence, pos_name, neg_name),
        "",
        markdown_table(
            "Escalation: is there a person or vehicle?", escalation, ", ".join(sorted(esc_positive)), "all other frames"
        ),
        "",
        f"p50 Decider wall time for this {len(questions)}-question call: {result['decider_wall_latency_ms_p50']:.0f} ms.",
    ]
    if args.freeze:
        frozen = freeze(rows, cache, args.label_col, positive, negative, ignore, esc_positive, presence, escalation)
        Path(args.freeze).write_text(json.dumps(frozen, indent=2))
        result["frozen"] = frozen
        md += ["", "### Frozen", "", "```json", json.dumps(frozen, indent=2), "```"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return result


if __name__ == "__main__":
    main()
