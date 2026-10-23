"""Freeze the v1 and v2 routing thresholds from the tuning set only; the one writer of results/thresholds.json.

v1 policy: difficulty-gated tiers, per frame, from the 9-question wording call.
  wording         highest AUC (presence ties: more negatives skipped at zero misses;
                  escalation ties: widest gap between classes)
  skip_below      highest 0.01 step with no tuning animal frame scoring below it
  small_above     midpoint of the gap between the highest empty frame Decider called clear
                  and the lowest clear animal frame above it
  escalate_above  midpoint of the gap between the highest non-person/vehicle frame and the
                  lowest person/vehicle frame

v2 policy: confidence band on P(animal) (difficulty not used), burst-aware skip.
  skip_below      highest 0.01 step at which no tuning animal burst has max P(animal) below it
  small_above     lowest 0.01 step from which every higher step keeps tuning frames at or above
                  it at least 95% animal
  escalate_above  unchanged from v1 (same question, same tuning frames)

Inputs: results/tune_answers.jsonl (the wording call on every tuning frame),
results/tune_answers_production.jsonl (the 3-question production call on every frame of
every tuning burst), data/tune.csv, data/tune_bursts.csv and results/label_audit.csv.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from patterns.presence_router.tune_questions import (
    STATE,
    all_questions,
    apply_audit,
    is_positive,
    load_cache,
    question_signature,
    read_manifest,
    score_of,
    score_wordings,
    split_scores,
    zero_miss_threshold,
)

RESULTS = ROOT / "results"
NON_ANIMAL = {"empty", "car", "person"}
PRECISION = 0.95
STEP = 0.01


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def grid(lo: float = 0.0, hi: float = 1.0) -> list[float]:
    return [round(i * STEP, 2) for i in range(int(lo / STEP), int(hi / STEP) + 1)]


def freeze_v1() -> dict:
    rows = read_manifest(ROOT / "data/tune.csv")
    apply_audit(rows, RESULTS / "label_audit.csv", "label")
    positive, negative, ignore, esc_positive = set(), {"empty"}, {"car", "person"}, {"car", "person"}
    cache = load_cache(RESULTS / "tune_answers.jsonl", question_signature(all_questions(), STATE))
    presence, escalation = score_wordings(rows, cache, "label", positive, negative, ignore, esc_positive)
    pk = max(presence, key=lambda r: (r["auc"], r["neg_below_zero_miss"]))["wording"]
    ek = max(escalation, key=lambda r: (r["auc"], (r["min_pos"] or 0) - r["max_neg"]))["wording"]
    pos, neg = split_scores(rows, cache, pk, "label", positive, negative, ignore)
    skip_below = zero_miss_threshold(pos, 0, STEP)
    clear = [
        (score_of(cache[r["file_name"]]["answers"], pk), is_positive(r["label"], positive, negative))
        for r in rows
        if r["label"] not in ignore and cache[r["file_name"]]["answers"]["difficulty"]["choice"] == "clear"
    ]
    clear_neg = max((s for s, p in clear if not p), default=0.0)
    clear_pos = min((s for s, p in clear if p and s > clear_neg), default=1.0)
    small_above = max(skip_below, round((clear_neg + clear_pos) / 2, 2))
    epos, eneg = split_scores(rows, cache, ek, "label", esc_positive, set())
    return {
        "skip_below": round(skip_below, 2),
        "small_above": small_above,
        "escalate_above": round((max(eneg) + min(epos)) / 2, 2),
        "chosen_wording": pk,
        "chosen_escalate_wording": ek,
        "method": (
            f"Tuning set only. Wording: highest AUC (escalation ties broken by the widest gap between classes). skip_below: highest {STEP} grid value with zero tuning animals scoring below it "
            f"(miss budget 0; lowest animal score {min(pos):.3f}). small_above: midpoint of the gap between the highest-scoring "
            f"empty frame Decider called clear ({clear_neg:.3f}) and the lowest-scoring clear animal frame above it ({clear_pos:.3f}). "
            f"escalate_above: midpoint of the gap between the highest non-person/vehicle frame ({max(eneg):.3f}) and the lowest "
            f"person/vehicle frame ({min(epos):.3f}), labels audited by eye where CCT marks people as empty."
        ),
        "n_tune": len(rows),
        "n_tune_animal": len(pos),
        "n_tune_empty": len(neg),
        "n_tune_person_or_vehicle": len(epos),
        "use_difficulty": True,
    }


def main() -> dict:
    v1 = freeze_v1()
    audit = {a["file_name"]: a["audited_label"] for a in read_csv(RESULTS / "label_audit.csv")}
    answers = {}
    for line in (RESULTS / "tune_answers_production.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            answers[row["file_name"]] = row["answers"]
    anchors = read_csv(ROOT / "data/tune.csv")
    bursts = read_csv(ROOT / "data/tune_bursts.csv")

    frames = []
    for r in anchors:
        label = audit.get(r["file_name"], r["label"])
        if label in ("car", "person"):
            continue
        frames.append((answers[r["file_name"]]["present"]["noul"], label != "empty"))
    small_above = None
    for t in reversed(grid(0.3, 1.0)):
        sel = [a for p, a in frames if p >= t]
        if sel and sum(sel) / len(sel) < PRECISION:
            break
        small_above = t
    gap_neg = max(p for p, a in frames if not a)
    gap_pos = min(p for p, a in frames if a and p > gap_neg)

    seq_label, seq_scores = {}, defaultdict(list)
    anchor_seq = {}
    for b in bursts:
        seq_scores[b["seq_id"]].append(answers[b["file_name"]]["present"]["noul"])
        if b["anchor"] == "1":
            anchor_seq[b["file_name"]] = b["seq_id"]
    for r in anchors:
        seq_label[anchor_seq[r["file_name"]]] = audit.get(r["file_name"], r["label"])
    animal_max = sorted(max(seq_scores[s]) for s, lab in seq_label.items() if lab not in NON_ANIMAL)
    empty_max = [max(seq_scores[s]) for s, lab in seq_label.items() if lab == "empty"]
    skip_below = int(animal_max[0] / STEP + 1e-9) * STEP
    frame_zero_miss = int(min(p for p, a in frames if a) / STEP + 1e-9) * STEP

    v2 = {
        "skip_below": round(skip_below, 2),
        "small_above": small_above,
        "escalate_above": v1["escalate_above"],
        "use_difficulty": False,
        "burst_aware": True,
        "chosen_wording": v1["chosen_wording"],
        "chosen_escalate_wording": v1["chosen_escalate_wording"],
        "method": (
            "Tuning set only (80 frames, 80 bursts, every burst frame asked with the production call). Policy: escalate if "
            "P(person/vehicle) >= escalate_above in any frame of the burst; skip only if max P(animal) over the burst < skip_below; "
            "small (Nova Lite) if the frame's P(animal) >= small_above; otherwise large (Nova Pro, Decider unsure). Difficulty is "
            f"still asked but not used. skip_below: highest {STEP} step at which no tuning animal burst has max P(animal) below it "
            f"(lowest animal-burst max {animal_max[0]:.3f}; the per-frame rule gives {frame_zero_miss:.2f}). small_above: lowest "
            f"{STEP} step from which tuning frames at or above every higher step are at least {PRECISION:.0%} animal (the midpoint-of-gap "
            f"rule would give {round((gap_neg + gap_pos) / 2, 2)}). escalate_above: unchanged from v1."
        ),
        "n_tune": len(anchors),
        "n_tune_bursts": len(seq_label),
        "n_tune_burst_frames": len(bursts),
        "n_tune_animal_bursts": len(animal_max),
        "n_tune_empty_bursts": len(empty_max),
        "tune_empty_bursts_skipped": sum(m < skip_below for m in empty_max),
    }
    out = {**v2, "v1": v1}
    (RESULTS / "thresholds.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
