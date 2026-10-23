"""Evaluate Trailgate on the frozen held-out set and the held-out vehicle set.

Decider is asked ONCE per frame with the production questions (questions.build_questions,
the exact call the router makes), and raw answers are cached to results/decider_answers.jsonl
so the run resumes. Frozen thresholds from results/thresholds.json are then applied with
PresenceRouterCore.decide. Nova baselines are merged in when results/nova_{lite,pro}.jsonl exist.

    .venv/bin/python scripts/evaluate.py            # ask Decider where not cached, then report
    .venv/bin/python scripts/evaluate.py --no-ask   # report from cache only
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

from patterns.presence_router.questions import build_questions
from patterns.presence_router.router import PresenceRouterCore, Thresholds, Tier
from patterns.presence_router.tune_questions import STATE, ask_all, auc, load_cache, question_signature, read_manifest
from trailgate import bedrock as nova
from trailgate import frames

RESULTS = ROOT / "results"
IMAGES = ROOT / "data/images"
NON_ANIMAL = {"empty", "car"}
VEHICLE = {"car"}
TIERS = [t.value for t in Tier]
LIGHTINGS = ["day", "night_ir"]
SWEEP = [round(i * 0.01, 2) for i in range(0, 101)]
ALERT_OUTPUT_TOKENS = 40


def pct(a: int, b: int) -> float | None:
    return round(100 * a / b, 1) if b else None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (pos - lo), 1)


def machine() -> str:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()) // 2**30
        return f"{chip}, {mem} GB, macOS {platform.mac_ver()[0]}, PyTorch MPS"
    except Exception:
        return platform.platform()


def route_frames(rows: list[dict], cache: dict, core: PresenceRouterCore) -> list[dict]:
    out = []
    for r in rows:
        c = cache[r["file_name"]]
        check = frames.check(IMAGES / r["file_name"])
        route = core.decide(c["answers"], c["wall_latency_ms"])
        tier = route.tier.value if check.ok else Tier.SKIP.value
        w, h = Image.open(IMAGES / r["file_name"]).size
        out.append({
            **r,
            "tier": tier,
            "present": route.present,
            "escalate": route.escalate,
            "difficulty": route.difficulty,
            "check": check.reason,
            "latency_ms": c["wall_latency_ms"],
            "width": w,
            "height": h,
            "is_animal": r["label"] not in NON_ANIMAL,
            "is_empty": r["label"] == "empty",
            "is_vehicle": r["label"] in VEHICLE,
        })
    return out


def tier_for(f: dict, t: Thresholds) -> str:
    if f["check"]:
        return "skip"
    if f["escalate"] is not None and f["escalate"] >= t.escalate_above:
        return "escalate"
    if f["present"] < t.skip_below:
        return "skip"
    if f["present"] >= t.small_above and f["difficulty"] == "clear":
        return "small"
    return "large"


def summary(frames_: list[dict]) -> dict:
    empties = [f for f in frames_ if f["is_empty"]]
    animals = [f for f in frames_ if f["is_animal"]]
    vehicles = [f for f in frames_ if f["is_vehicle"]]
    non_vehicle = [f for f in frames_ if not f["is_vehicle"]]
    lat = [f["latency_ms"] for f in frames_]
    return {
        "n_frames": len(frames_),
        "n_empty": len(empties),
        "n_animal": len(animals),
        "n_vehicle": len(vehicles),
        "empty_skipped": sum(f["tier"] == "skip" for f in empties),
        "empty_skipped_pct": pct(sum(f["tier"] == "skip" for f in empties), len(empties)),
        "animals_missed": sum(f["tier"] == "skip" for f in animals),
        "animals_missed_pct": pct(sum(f["tier"] == "skip" for f in animals), len(animals)),
        "frames_skipped": sum(f["tier"] == "skip" for f in frames_),
        "frames_skipped_pct": pct(sum(f["tier"] == "skip" for f in frames_), len(frames_)),
        "tiers": {t: sum(f["tier"] == t for f in frames_) for t in TIERS},
        "animal_tiers": {t: sum(f["tier"] == t for f in animals) for t in TIERS},
        "empty_tiers": {t: sum(f["tier"] == t for f in empties) for t in TIERS},
        "vehicles_caught": sum(f["tier"] == "escalate" for f in vehicles),
        "false_alerts": sum(f["tier"] == "escalate" for f in non_vehicle),
        "false_alerts_on_empty": sum(f["tier"] == "escalate" for f in empties),
        "presence_auc_animal_vs_empty": round(auc([f["present"] for f in animals], [f["present"] for f in empties]), 4)
        if animals and empties else None,
        "mean_present_animal": round(sum(f["present"] for f in animals) / len(animals), 3) if animals else None,
        "mean_present_empty": round(sum(f["present"] for f in empties) / len(empties), 3) if empties else None,
        "latency_ms_p50": quantile(lat, 0.5),
        "latency_ms_p90": quantile(lat, 0.9),
    }


def sweep(frames_: list[dict], t: Thresholds) -> list[dict]:
    rows = []
    for s in SWEEP:
        tt = Thresholds(skip_below=s, small_above=t.small_above, escalate_above=t.escalate_above)
        tiers = [tier_for(f, tt) for f in frames_]
        rows.append({
            "skip_below": s,
            "frames_skipped": sum(x == "skip" for x in tiers),
            "frames_skipped_pct": pct(sum(x == "skip" for x in tiers), len(frames_)),
            "empty_skipped": sum(x == "skip" and f["is_empty"] for x, f in zip(tiers, frames_)),
            "animals_missed": sum(x == "skip" and f["is_animal"] for x, f in zip(tiers, frames_)),
            "animals_missed_pct": pct(sum(x == "skip" and f["is_animal"] for x, f in zip(tiers, frames_)),
                                      sum(f["is_animal"] for f in frames_)),
        })
    return rows


def reliability(frames_: list[dict], bins: int = 10) -> list[dict]:
    pool = [f for f in frames_ if f["is_animal"] or f["is_empty"]]
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        inside = [f for f in pool if lo <= f["present"] < hi or (i == bins - 1 and f["present"] == 1.0)]
        out.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "lo": lo,
            "hi": hi,
            "n": len(inside),
            "n_animal": sum(f["is_animal"] for f in inside),
            "mean_present": round(sum(f["present"] for f in inside) / len(inside), 3) if inside else None,
            "observed_animal_rate": round(sum(f["is_animal"] for f in inside) / len(inside), 3) if inside else None,
        })
    return out


def dataset_stats() -> dict | None:
    path = ROOT / "data/caltech_images_20210113.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    per: dict = {}
    for a in d["annotations"]:
        per.setdefault(a["image_id"], set()).add(cats[a["category_id"]])
    empty = sum(1 for i in d["images"] if per.get(i["id"]) == {"empty"})
    return {"source": path.name, "n_images": len(d["images"]), "n_empty": empty,
            "empty_pct": pct(empty, len(d["images"])), "has_person_class": "person" in cats.values() or "human" in cats.values()}


def frame_costs(f: dict) -> dict:
    w, h = f["width"], f["height"]
    lite = nova.baseline_cost("small", w, h)
    pro = nova.baseline_cost("large", w, h)
    alert_in = nova.estimate_image_tokens(w, h) + nova.estimate_text_tokens(nova.ALERT_PROMPT)
    alert = nova.cost(nova.model_for("small"), alert_in, ALERT_OUTPUT_TOKENS)
    routed = {"skip": 0.0, "small": lite, "large": pro, "escalate": alert + pro}[f["tier"]]
    gated_lite = {"skip": 0.0, "small": lite, "large": lite, "escalate": alert + lite}[f["tier"]]
    return {"lite": lite, "pro": pro, "trailgate": routed, "gated_lite": gated_lite}


def cost_model(frames_: list[dict]) -> dict:
    n = len(frames_)
    totals = Counter()
    for f in frames_:
        totals.update(frame_costs(f))
    scale = 10000 / n
    w, h = Counter((f["width"], f["height"]) for f in frames_).most_common(1)[0][0]
    tokens = nova.estimate_image_tokens(w, h) + nova.estimate_text_tokens(nova.LABEL_TEXT)
    return {
        "basis": f"held-out set, {n} frames, natural empty/animal mix; scaled to 10,000 frames",
        "status": "estimated from published token formula and prices; no Bedrock call has been measured yet",
        "per_10k_usd": {
            "nova_lite_every_frame": round(totals["lite"] * scale, 4),
            "nova_pro_every_frame": round(totals["pro"] * scale, 4),
            "trailgate": round(totals["trailgate"] * scale, 4),
            "trailgate_lite_only": round(totals["gated_lite"] * scale, 4),
        },
        "trailgate_vs_lite_pct": pct(totals["trailgate"], totals["lite"]),
        "trailgate_vs_pro_pct": pct(totals["trailgate"], totals["pro"]),
        "trailgate_lite_only_vs_lite_pct": pct(totals["gated_lite"], totals["lite"]),
        "trailgate_routing": "skip: $0; small: Nova Lite label; large: Nova Pro label; escalate: Nova Lite alert + Nova Pro label (pipeline default label_escalations=True)",
        "label_call_tokens_typical_frame": {"input": tokens, "output": nova.LABEL_OUTPUT_TOKENS, "frame": f"{w}x{h}"},
        "image_tokens_table": [list(r) for r in nova.NOVA_IMAGE_TOKENS],
        "prices_per_1k_tokens_usd": {k: {"input": v[0], "output": v[1]} for k, v in nova.PRICES.items()},
        "sources": [nova.PRICES_SOURCE, nova.IMAGE_TOKENS_SOURCE],
        "decider_cost_per_call_usd": 0.0,
        "decider_machine": machine(),
    }


def load_nova(name: str) -> dict:
    path = RESULTS / f"nova_{name}.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if not row.get("error"):
                out[row["file_name"]] = row
    return out


def species_eval(frames_: list[dict], predict) -> dict:
    scored = [(f, predict(f)) for f in frames_]
    scored = [(f, p) for f, p in scored if p is not None]
    animals = [(f, p) for f, p in scored if f["is_animal"]]
    empties = [(f, p) for f, p in scored if f["is_empty"]]
    return {
        "n_scored": len(scored),
        "label_accuracy_all": pct(sum(p == f["label"] for f, p in scored), len(scored)),
        "species_accuracy_on_animals": pct(sum(p == f["label"] for f, p in animals), len(animals)),
        "empty_correct": sum(p == "empty" for f, p in empties),
        "empty_correct_pct": pct(sum(p == "empty" for f, p in empties), len(empties)),
        "animals_said_empty": sum(p == "empty" for f, p in animals),
    }


def nova_baselines(frames_: list[dict]) -> dict | None:
    lite, pro = load_nova("lite"), load_nova("pro")
    if not lite and not pro:
        return None
    out = {}
    for name, rows in (("nova_lite", lite), ("nova_pro", pro)):
        if not rows:
            continue
        calls = list(rows.values())
        out[name] = {
            **species_eval(frames_, lambda f: rows.get(f["file_name"], {}).get("species")),
            "measured_cost_usd": round(sum(c["cost_usd"] for c in calls), 6),
            "measured_cost_per_10k_usd": round(sum(c["cost_usd"] for c in calls) / len(calls) * 10000, 4),
            "latency_ms_p50": quantile([c["latency_ms"] for c in calls], 0.5),
            "mean_input_tokens": round(sum(c["input_tokens"] for c in calls) / len(calls), 1),
        }
    if lite and pro:
        def trail(f):
            if f["tier"] == "skip":
                return "empty"
            src = lite if f["tier"] == "small" else pro
            return src.get(f["file_name"], {}).get("species")
        covered = [f for f in frames_ if f["tier"] == "skip" or trail(f) is not None]
        cost = 0.0
        for f in covered:
            if f["tier"] == "small":
                cost += lite[f["file_name"]]["cost_usd"]
            elif f["tier"] in ("large", "escalate"):
                cost += pro[f["file_name"]]["cost_usd"]
        out["trailgate"] = {
            **species_eval(frames_, trail),
            "measured_label_cost_per_10k_usd": round(cost / len(covered) * 10000, 4) if covered else None,
            "note": "Trailgate species = Nova Lite answer on small-tier frames, Nova Pro on large/escalate, 'empty' on skip; alert-call cost not included",
        }
    return out


def fmt(v, suffix=""):
    return "n/a" if v is None else f"{v}{suffix}"


def write_markdown(m: dict) -> str:
    h, t = m["heldout"], m["thresholds"]
    lines = [
        "# Trailgate evaluation",
        "",
        f"Frozen thresholds from the tuning set (`results/thresholds.json`): skip below {t['skip_below']}, "
        f"small at or above {t['small_above']} when difficulty is clear, escalate at or above {t['escalate_above']}. "
        f"Presence wording `{t['chosen_wording']}`. Rule: {t['method']}",
        "",
        f"Held-out set: {h['all']['n_frames']} Caltech Camera Traps frames from camera locations never used for tuning, "
        f"natural mix ({h['all']['n_empty']} empty, {h['all']['n_animal']} animal, {h['all']['n_vehicle']} car). "
        f"Vehicle set: {m['vehicle_set']['n_frames']} `car` frames from held-out locations. "
        "One Decider call per frame. Small set; 0 means no errors on this set, not solved.",
        "",
        (f"Full CCT label file (`{m['dataset']['source']}`): {m['dataset']['n_empty']:,} of {m['dataset']['n_images']:,} images are empty "
         f"({m['dataset']['empty_pct']}%; lila.science's summary says about 70%). The label set has no person class."
         if m.get("dataset") else ""),
        "",
        "## Headline (held-out set)",
        "",
        "| Subset | Frames | Empty correctly skipped | Animals missed (routed to skip) | Frames skipped | Vehicles caught | Alerts on frames without a car label | Presence AUC | p50 / p90 Decider latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, name in (("all", "All"), ("day", "Day"), ("night_ir", "Night (infrared)")):
        s = h[key]
        lines.append(
            f"| {name} | {s['n_frames']} | {s['empty_skipped']}/{s['n_empty']} ({fmt(s['empty_skipped_pct'], '%')}) | "
            f"**{s['animals_missed']}/{s['n_animal']}** ({fmt(s['animals_missed_pct'], '%')}) | "
            f"{s['frames_skipped']} ({fmt(s['frames_skipped_pct'], '%')}) | {s['vehicles_caught']}/{s['n_vehicle']} | "
            f"{s['false_alerts']} | {fmt(s['presence_auc_animal_vs_empty'])} | {fmt(s['latency_ms_p50'])} / {fmt(s['latency_ms_p90'])} ms |"
        )
    w = m.get("tune_reference")
    if w:
        lines += ["", f"Tuning set vs held-out, same wording and threshold: presence AUC {w['auc']} on tuning vs {h['all']['presence_auc_animal_vs_empty']} "
                  f"held-out; empty frames below skip_below {w['neg_below_zero_miss'] * 100:.1f}% on tuning (0 animals below) vs "
                  f"{h['all']['empty_skipped_pct']}% held-out ({h['all']['animals_missed']} animals below). The threshold did not transfer without misses."]
    v = m["vehicles"]
    lines += [
        "",
        f"**Vehicles caught:** {v['caught']}/{v['n']} ({fmt(v['caught_pct'], '%')}) across the vehicle set "
        f"({m['vehicle_set']['vehicles_caught']}/{m['vehicle_set']['n_frames']}) and held-out `car` frames "
        f"({h['all']['vehicles_caught']}/{h['all']['n_vehicle']}). False alerts on held-out non-vehicle frames: "
        f"{h['all']['false_alerts']} ({h['all']['false_alerts_on_empty']} of them labelled empty).",
        "",
        "## Tier distribution (held-out)",
        "",
        "| Frames | " + " | ".join(TIERS) + " |",
        "|---|" + "---|" * len(TIERS),
        "| All | " + " | ".join(str(h["all"]["tiers"][x]) for x in TIERS) + " |",
        "| Labelled empty | " + " | ".join(str(h["all"]["empty_tiers"][x]) for x in TIERS) + " |",
        "| Labelled animal | " + " | ".join(str(h["all"]["animal_tiers"][x]) for x in TIERS) + " |",
        "",
        "## The safety knob: skip threshold against animals missed (held-out)",
        "",
        "Other thresholds held at their frozen values. The frozen point is in bold.",
        "",
        "| skip_below | Frames skipped | Empty skipped | Animals missed |",
        "|---|---|---|---|",
    ]
    marks = {round(t["skip_below"], 2)}
    for r in m["sweep"]:
        if round(r["skip_below"] * 100) % 5 == 0 or r["skip_below"] in marks:
            row = f"| {r['skip_below']:.2f} | {r['frames_skipped']} ({r['frames_skipped_pct']}%) | {r['empty_skipped']}/{h['all']['n_empty']} | {r['animals_missed']}/{h['all']['n_animal']} ({r['animals_missed_pct']}%) |"
            if r["skip_below"] in marks:
                row = "| " + " | ".join(f"**{c.strip()}**" for c in row.strip("|").split("|")) + " |"
            lines.append(row)
    lines += [
        "",
        "## Reliability (held-out, animal vs empty frames)",
        "",
        "| P(animal) bin | Frames | Animal frames | Mean P(animal) | Observed animal rate |",
        "|---|---|---|---|---|",
    ]
    for b in m["reliability"]:
        lines.append(f"| {b['bin']} | {b['n']} | {b['n_animal']} | {fmt(b['mean_present'])} | {fmt(b['observed_animal_rate'])} |")
    c = m["cost"]
    tok = c["label_call_tokens_typical_frame"]
    lines += [
        "",
        "## Cost per 10,000 frames",
        "",
        f"**{c['status']}.** Basis: {c['basis']}.",
        "",
        "| System | $ per 10,000 frames |",
        "|---|---|",
        f"| Nova Lite on every frame | ${c['per_10k_usd']['nova_lite_every_frame']:.2f} |",
        f"| Nova Pro on every frame | ${c['per_10k_usd']['nova_pro_every_frame']:.2f} |",
        f"| Trailgate (Decider routes: skip / Nova Lite / Nova Pro / alert) | ${c['per_10k_usd']['trailgate']:.2f} |",
        f"| Trailgate, Lite-only variant (Decider skips; every other frame to Nova Lite) | ${c['per_10k_usd']['trailgate_lite_only']:.2f} |",
        "",
        f"Trailgate as routed costs {c['trailgate_vs_pro_pct']}% of Nova Pro on every frame and {c['trailgate_vs_lite_pct']}% of Nova Lite on every frame "
        f"(most frames it does not skip go to Nova Pro). The Lite-only variant costs {c['trailgate_lite_only_vs_lite_pct']}% of Nova Lite on every frame. "
        f"Routing cost: {c['trailgate_routing']}.",
        "",
        f"Per label call on a {tok['frame']} frame: ~{tok['input']} input tokens (image estimate from the closest documented resolution "
        f"plus the prompt) and {tok['output']} output tokens. On-demand prices per 1K tokens (us-east-1): "
        + "; ".join(f"{k} ${v['input']} in / ${v['output']} out" for k, v in c["prices_per_1k_tokens_usd"].items()) + ".",
        "",
        f"Decider runs locally: $0 per call, on {c['decider_machine']}.",
        "",
        "Sources:",
        "",
    ]
    lines += [f"- {s}" for s in c["sources"]]
    if m.get("nova"):
        lines += ["", "## Measured Bedrock baselines (held-out)", "",
                  "| System | Frames scored | Species accuracy on animal frames | Empty frames called empty | Animals called empty | Measured $ per 10,000 |",
                  "|---|---|---|---|---|---|"]
        for name, s in m["nova"].items():
            cost = s.get("measured_cost_per_10k_usd", s.get("measured_label_cost_per_10k_usd"))
            lines.append(f"| {name} | {s['n_scored']} | {fmt(s['species_accuracy_on_animals'], '%')} | {s['empty_correct']} ({fmt(s['empty_correct_pct'], '%')}) | {s['animals_said_empty']} | {fmt(cost)} |")
    else:
        lines += ["", "Bedrock baselines: not yet measured (Bedrock on-demand quota on this account was 0 at evaluation time; "
                  "`scripts/run_nova_baseline.py` fills `results/nova_{lite,pro}.jsonl` and this script merges them)."]
    a = m["visual_audit"]
    lines += [
        "",
        "## Visual check of the errors",
        "",
        f"Scope: {a['scope']}. Details in `results/label_audit.csv`.",
        "",
        f"- **False alerts:** {a['false_alerts_with_person']} of the {a['false_alerts_checked']} held-out frames escalated without a `car` label show people. "
        "CCT has no person class, so people are labelled `empty`; these alerts are correct.",
        f"- **Missed animals ({len(m['missed'])}):** visible to the eye: {a['missed_animals_visible'].get('yes', 0)}; faint (a shape or eye shine): "
        f"{a['missed_animals_visible'].get('faint', 0)}; nothing visible: {a['missed_animals_visible'].get('no', 0)}. "
        f"{a['missed_animals_later_burst_frame']} of them are frame 2 or 3 of a camera burst (read from the on-image stamp), "
        "consistent with one label per burst: the animal may already have left the frame.",
        f"- **Missed vehicles ({len(a['missed_vehicles'])}):** "
        + "; ".join(f"`{v['file_name'][:8]}` P={v['escalate']:.2f}, {v['note']}" for v in a["missed_vehicles"]) + ".",
    ]
    b = m.get("batching_check")
    if b:
        lines += ["", f"Batching check: on {b['n_frames']} tuning frames, asking the same wording in a 9-question call and in the 3-question "
                  f"production call changed P(animal) by at most {b['present_max_abs_diff']} and P(person/vehicle) by at most "
                  f"{b['escalate_max_abs_diff']}; difficulty flipped on {b['difficulty_flips']} frames."]
    lines += ["", f"Missed animals with thumbnails: `results/missed.md` ({len(m['missed'])} frames).", ""]
    return "\n".join(lines)


def load_audit() -> dict:
    path = RESULTS / "label_audit.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {a["file_name"]: a for a in csv.DictReader(f)}


def audit_summary(hf: list[dict], vf: list[dict], missed: list[dict], audit: dict) -> dict:
    alerts = [f for f in hf if f["tier"] == "escalate" and not f["is_vehicle"]]
    missed_veh = [f for f in hf + vf if f["is_vehicle"] and f["tier"] != "escalate"]
    seen = lambda fs: Counter(audit.get(f["file_name"], {}).get("visible", "unchecked") for f in fs)
    return {
        "scope": "only frames the router got wrong by CCT labels were checked by eye; the rest of the set was not audited",
        "false_alerts_checked": len(alerts),
        "false_alerts_with_person": sum(audit.get(f["file_name"], {}).get("audited_label") == "person" for f in alerts),
        "missed_animals_visible": dict(seen(missed)),
        "missed_vehicles_visible": dict(seen(missed_veh)),
        "missed_animals_later_burst_frame": sum(any(b in audit.get(f["file_name"], {}).get("note", "") for b in ("burst 2/", "burst 3/"))
                                                for f in missed),
        "missed_vehicles": [{"file_name": f["file_name"], "escalate": f["escalate"], "tier": f["tier"],
                             "note": audit.get(f["file_name"], {}).get("note")} for f in missed_veh],
    }


def batching_check() -> dict | None:
    batched, single = RESULTS / "tune_answers.jsonl", RESULTS / "tune_answers_production.jsonl"
    if not (batched.exists() and single.exists()):
        return None
    a = {json.loads(l)["file_name"]: json.loads(l)["answers"] for l in batched.read_text().splitlines() if l.strip()}
    b = {json.loads(l)["file_name"]: json.loads(l)["answers"] for l in single.read_text().splitlines() if l.strip()}
    common = [k for k in b if k in a]
    dp = [abs(a[k]["p_criteria"]["noul"] - b[k]["present"]["noul"]) for k in common]
    de = [abs(a[k]["e_criteria"]["noul"] - b[k]["escalate"]["noul"]) for k in common]
    return {
        "n_frames": len(common),
        "present_max_abs_diff": round(max(dp), 4),
        "escalate_max_abs_diff": round(max(de), 4),
        "difficulty_flips": sum(a[k]["difficulty"]["choice"] != b[k]["difficulty"]["choice"] for k in common),
        "note": "same wording asked in a 9-question batch (tuning) and in the 3-question production call",
    }


def save_thumbnails(missed: list[dict], audit: dict | None = None) -> None:
    audit = audit or {}
    out = RESULTS / "missed"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Animal frames Trailgate skipped (held-out)", "",
             "Every held-out frame labelled with an animal that the frozen router sent to `skip`.", "",
             "\"Visible to the eye\" is a manual check of the full-size frame: yes, faint (a shape or eye shine that may be the animal), or no.", "",
             "| Thumbnail | Label | Lighting | P(animal) | Visible to the eye | Note | Location | Frame |", "|---|---|---|---|---|---|---|---|"]
    for f in sorted(missed, key=lambda x: x["present"]):
        img = Image.open(IMAGES / f["file_name"])
        img.thumbnail((400, 400))
        img.convert("RGB").save(out / f["file_name"], quality=85)
        a = audit.get(f["file_name"], {})
        lines.append(f"| ![{f['label']}](missed/{f['file_name']}) | {f['label']} | {f['lighting']} | {f['present']:.3f} | "
                     f"{a.get('visible', 'not checked')} | {a.get('note', '')} | {f['location']} | `{f['file_name']}` |")
    (RESULTS / "missed.md").write_text("\n".join(lines) + "\n")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-ask", action="store_true")
    ap.add_argument("--ask-only", action="store_true")
    ap.add_argument("--cache", default=str(RESULTS / "decider_answers.jsonl"))
    args = ap.parse_args(argv)

    questions = build_questions()
    heldout = read_manifest(ROOT / "data/manifests/heldout.csv")
    vehicle = read_manifest(ROOT / "data/manifests/vehicle.csv")
    if args.no_ask:
        cache = load_cache(Path(args.cache), question_signature(questions, STATE))
    else:
        cache = ask_all(heldout + vehicle, questions, Path(args.cache), IMAGES)
    if args.ask_only:
        return {}
    tfile = json.loads((RESULTS / "thresholds.json").read_text())
    t = Thresholds(tfile["skip_below"], tfile["small_above"], tfile["escalate_above"])
    heldout = [r for r in heldout if r["file_name"] in cache]
    vehicle = [r for r in vehicle if r["file_name"] in cache]
    core = PresenceRouterCore(thresholds=t)
    hf = route_frames(heldout, cache, core)
    vf = route_frames(vehicle, cache, core)

    vs = summary(vf)
    hs = {"all": summary(hf), **{l: summary([f for f in hf if f["lighting"] == l]) for l in LIGHTINGS}}
    caught = hs["all"]["vehicles_caught"] + vs["vehicles_caught"]
    n_veh = hs["all"]["n_vehicle"] + vs["n_vehicle"]
    missed = [f for f in hf if f["is_animal"] and f["tier"] == "skip"]
    lat_all = [f["latency_ms"] for f in hf + vf]
    audit = load_audit()
    metrics = {
        "thresholds": tfile,
        "questions": questions,
        "state": STATE,
        "heldout": hs,
        "vehicle_set": vs,
        "vehicle_set_by_lighting": {l: summary([f for f in vf if f["lighting"] == l]) for l in LIGHTINGS},
        "vehicles": {"caught": caught, "n": n_veh, "caught_pct": pct(caught, n_veh)},
        "decider_latency_ms_all_calls": {"n": len(lat_all), "p50": quantile(lat_all, 0.5), "p90": quantile(lat_all, 0.9),
                                         "note": "wall clock around one HTTP call with the 3 production questions"},
        "sweep": sweep(hf, t),
        "reliability": reliability(hf),
        "reliability_by_lighting": {l: reliability([f for f in hf if f["lighting"] == l]) for l in LIGHTINGS},
        "cost": cost_model(hf),
        "nova": nova_baselines(hf),
        "missed": [{k: f[k] for k in ("file_name", "label", "lighting", "location", "present", "difficulty")} for f in missed],
        "dataset": dataset_stats(),
        "tune_reference": next((r for r in json.loads((RESULTS / "wording.json").read_text())["presence"]
                                if r["wording"] == tfile["chosen_wording"]), None) if (RESULTS / "wording.json").exists() else None,
        "visual_audit": audit_summary(hf, vf, missed, audit),
        "batching_check": batching_check(),
        "code_check_skips": [{k: f[k] for k in ("file_name", "label", "check")} for f in hf + vf if f["check"]],
        "label_noise_candidates": {
            "empty_with_high_escalate": [{k: f[k] for k in ("file_name", "lighting", "present", "escalate")}
                                         for f in sorted(hf, key=lambda x: -(x["escalate"] or 0))
                                         if f["is_empty"] and (f["escalate"] or 0) >= t.escalate_above],
            "empty_with_high_present": [{k: f[k] for k in ("file_name", "lighting", "present", "escalate")}
                                        for f in sorted(hf, key=lambda x: -x["present"]) if f["is_empty"] and f["present"] >= 0.95],
        },
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS / "metrics.md").write_text(write_markdown(metrics))
    (RESULTS / "eval_frames.jsonl").write_text("\n".join(json.dumps(f) for f in hf + vf) + "\n")
    save_thumbnails(missed, audit)
    from charts import draw_all
    draw_all(metrics, hf)
    print(write_markdown(metrics))
    return metrics


if __name__ == "__main__":
    main()
