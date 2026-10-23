"""Freeze the v2 routing thresholds from the tuning set only, keeping v1 alongside.

v2 policy: confidence band on P(animal) (difficulty not used), burst-aware skip.
  skip_below      highest 0.01 step at which no tuning animal burst has max P(animal) below it
  small_above     lowest 0.01 step from which every higher step keeps tuning frames at or above
                  it at least 95% animal
  escalate_above  unchanged from v1 (same question, same tuning frames)

Inputs: results/tune_answers_production.jsonl (the 3-question production call on every frame of
every tuning burst), data/manifests/tune_bursts.csv, results/label_audit.csv, and the v1 values
in results/thresholds.json.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
NON_ANIMAL = {"empty", "car", "person"}
PRECISION = 0.95
STEP = 0.01


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def grid(lo: float = 0.0, hi: float = 1.0) -> list[float]:
    return [round(i * STEP, 2) for i in range(int(lo / STEP), int(hi / STEP) + 1)]


def main() -> dict:
    current = json.loads((RESULTS / "thresholds.json").read_text())
    v1 = current.get("v1") or {**{k: v for k, v in current.items() if k != "v2"}, "use_difficulty": True}
    audit = {a["file_name"]: a["audited_label"] for a in read_csv(RESULTS / "label_audit.csv")}
    answers = {}
    for line in (RESULTS / "tune_answers_production.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            answers[row["file_name"]] = row["answers"]
    anchors = read_csv(ROOT / "data/manifests/tune.csv")
    bursts = read_csv(ROOT / "data/manifests/tune_bursts.csv")

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
    animal_max = sorted(max(seq_scores[s]) for s, l in seq_label.items() if l not in NON_ANIMAL)
    empty_max = [max(seq_scores[s]) for s, l in seq_label.items() if l == "empty"]
    skip_below = int(animal_max[0] / STEP + 1e-9) * STEP
    frame_zero_miss = int(min(p for p, a in frames if a) / STEP + 1e-9) * STEP

    v2 = {
        "skip_below": round(skip_below, 2),
        "small_above": small_above,
        "escalate_above": v1["escalate_above"],
        "use_difficulty": False,
        "burst_aware": True,
        "chosen_wording": v1.get("chosen_wording", "p_criteria"),
        "chosen_escalate_wording": v1.get("chosen_escalate_wording", "e_criteria"),
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
    out = {**{k: v for k, v in v2.items()}, "v1": v1}
    (RESULTS / "thresholds.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
