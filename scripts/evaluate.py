"""Evaluate Trailgate on the frozen held-out set: v1 (as committed) and v2 (confidence band + burst).

Decider is asked ONCE per frame with the production questions (questions.build_questions, the
exact call the router makes) on every held-out frame, every other frame of its camera burst
(data/manifests/heldout_bursts.csv) and the vehicle set. Raw answers are cached to
results/decider_answers.jsonl so the run resumes. Frozen thresholds from results/thresholds.json
(v2 at the top level, v1 under "v1") are applied with PresenceRouterCore.decide and decide_burst.
Nova baselines are merged in when results/nova_{lite,pro}.jsonl exist.

    .venv/bin/python scripts/evaluate.py            # ask Decider where not cached, then report
    .venv/bin/python scripts/evaluate.py --no-ask   # report from cache only
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

from patterns.presence_router.questions import build_questions
from patterns.presence_router.router import DeciderClient, PresenceRouterCore, Thresholds, Tier
from patterns.presence_router.tune_questions import STATE, ask_all, auc, load_cache, question_signature, read_manifest
from trailgate import bedrock as nova
from trailgate import frames

RESULTS = ROOT / "results"
IMAGES = ROOT / "data/images"
NON_ANIMAL = {"empty", "car"}
VEHICLE = {"car"}
TIERS = [t.value for t in Tier]
LIGHTINGS = ["day", "night_ir"]
LIGHT_NAME = {"all": "All", "day": "Day", "night_ir": "Night (infrared)"}
SWEEP = [round(i * 0.01, 2) for i in range(0, 101)]
ALERT_OUTPUT_TOKENS = 40
POLICY_NAMES = {
    "v1": "v1: difficulty-gated tiers, per frame (as committed)",
    "v2_frame": "v2 band only: confidence band on P(animal), per frame (ablation, v2 thresholds)",
    "v2": "v2: confidence band + burst context",
}


def pct(a: int, b: int) -> float | None:
    return round(100 * a / b, 1) if b else None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (pos - lo), 1)


def fmt(v, suffix=""):
    return "n/a" if v is None else f"{v}{suffix}"


def machine() -> str:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()) // 2**30
        return f"{chip}, {mem} GB, macOS {platform.mac_ver()[0]}, PyTorch MPS"
    except Exception:
        return platform.platform()


def thresholds_of(d: dict) -> Thresholds:
    return Thresholds(float(d["skip_below"]), float(d["small_above"]), float(d["escalate_above"]), bool(d.get("use_difficulty", False)))


def frame_rows(rows: list[dict], cache: dict, kind: str) -> list[dict]:
    out = []
    for r in rows:
        c = cache[r["file_name"]]
        a = c["answers"]
        w, h = Image.open(IMAGES / r["file_name"]).size
        out.append({
            **r,
            "kind": kind,
            "seq_id": r.get("seq_id") or r["file_name"],
            "anchor": r.get("anchor", "1") == "1",
            "answers": a,
            "present": a["present"]["noul"],
            "escalate": a["escalate"]["noul"],
            "difficulty": a["difficulty"]["choice"],
            "p_hard": a["difficulty"].get("probabilities", {}).get("hard"),
            "check": frames.check(IMAGES / r["file_name"]).reason,
            "latency_ms": c["wall_latency_ms"],
            "width": w,
            "height": h,
            "is_animal": r["label"] not in NON_ANIMAL,
            "is_empty": r["label"] == "empty",
            "is_vehicle": r["label"] in VEHICLE,
        })
    return out


def route_all(rows: list[dict], t: Thresholds, burst: bool) -> list[str]:
    core = PresenceRouterCore(thresholds=t, client=DeciderClient())
    routes = [core.decide(r["answers"], r["latency_ms"]) for r in rows]
    if burst:
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[r["seq_id"]].append(i)
        for idx in groups.values():
            for i, route in zip(idx, core.decide_burst([routes[i] for i in idx])):
                routes[i] = route
    return [Tier.SKIP.value if r["check"] else route.tier.value for r, route in zip(rows, routes)]


def summary(fs: list[dict]) -> dict:
    empties = [f for f in fs if f["is_empty"]]
    animals = [f for f in fs if f["is_animal"]]
    vehicles = [f for f in fs if f["is_vehicle"]]
    non_vehicle = [f for f in fs if not f["is_vehicle"]]
    lat = [f["latency_ms"] for f in fs]
    skip = lambda group: sum(f["tier"] == "skip" for f in group)
    return {
        "n_frames": len(fs),
        "n_empty": len(empties),
        "n_animal": len(animals),
        "n_vehicle": len(vehicles),
        "empty_skipped": skip(empties),
        "empty_skipped_pct": pct(skip(empties), len(empties)),
        "animals_missed": skip(animals),
        "animals_missed_pct": pct(skip(animals), len(animals)),
        "frames_skipped": skip(fs),
        "frames_skipped_pct": pct(skip(fs), len(fs)),
        "tiers": {t: sum(f["tier"] == t for f in fs) for t in TIERS},
        "animal_tiers": {t: sum(f["tier"] == t for f in animals) for t in TIERS},
        "empty_tiers": {t: sum(f["tier"] == t for f in empties) for t in TIERS},
        "vehicles_caught": sum(f["tier"] == "escalate" for f in vehicles),
        "false_alerts": sum(f["tier"] == "escalate" for f in non_vehicle),
        "false_alerts_on_empty": sum(f["tier"] == "escalate" for f in empties),
        "presence_auc_animal_vs_empty": round(auc([f["present"] for f in animals], [f["present"] for f in empties]), 4)
        if animals and empties else None,
        "latency_ms_p50": quantile(lat, 0.5),
        "latency_ms_p90": quantile(lat, 0.9),
    }


def sequence_summary(burst_frames: list[dict], seq_label: dict) -> dict:
    groups = defaultdict(list)
    for f in burst_frames:
        groups[f["seq_id"]].append(f)
    out = {}
    for key in ("all", *LIGHTINGS):
        seqs = {s: g for s, g in groups.items() if key == "all" or seq_lighting(g) == key}
        empty = [g for s, g in seqs.items() if seq_label[s] == "empty"]
        animal = [g for s, g in seqs.items() if seq_label[s] not in NON_ANIMAL]
        vehicle = [g for s, g in seqs.items() if seq_label[s] in VEHICLE]
        all_skipped = lambda g: all(f["tier"] == "skip" for f in g)
        out[key] = {
            "n_sequences": len(seqs),
            "n_frames": sum(len(g) for g in seqs.values()),
            "frames_skipped": sum(f["tier"] == "skip" for g in seqs.values() for f in g),
            "n_empty": len(empty),
            "empty_skipped": sum(all_skipped(g) for g in empty),
            "empty_skipped_pct": pct(sum(all_skipped(g) for g in empty), len(empty)),
            "n_animal": len(animal),
            "animals_missed": sum(all_skipped(g) for g in animal),
            "animals_missed_pct": pct(sum(all_skipped(g) for g in animal), len(animal)),
            "animal_partly_skipped": sum(any(f["tier"] == "skip" for f in g) and not all_skipped(g) for g in animal),
            "n_vehicle": len(vehicle),
            "vehicles_caught": sum(any(f["tier"] == "escalate" for f in g) for g in vehicle),
        }
    return out


def seq_lighting(group: list[dict]) -> str:
    anchors = [f for f in group if f["anchor"]] or group
    return anchors[0]["lighting"]


def sweep(burst_frames: list[dict], t: Thresholds, burst: bool) -> list[dict]:
    rows = []
    n_animal = sum(f["is_animal"] for f in burst_frames if f["anchor"])
    for s in SWEEP:
        tt = Thresholds(s, t.small_above, t.escalate_above, t.use_difficulty)
        tiers = route_all(burst_frames, tt, burst)
        anchors = [(x, f) for x, f in zip(tiers, burst_frames) if f["anchor"]]
        rows.append({
            "skip_below": s,
            "frames_skipped": sum(x == "skip" for x, _ in anchors),
            "frames_skipped_pct": pct(sum(x == "skip" for x, _ in anchors), len(anchors)),
            "empty_skipped": sum(x == "skip" and f["is_empty"] for x, f in anchors),
            "animals_missed": sum(x == "skip" and f["is_animal"] for x, f in anchors),
            "animals_missed_pct": pct(sum(x == "skip" and f["is_animal"] for x, f in anchors), n_animal),
        })
    return rows


def reliability(fs: list[dict], bins: int = 10) -> list[dict]:
    pool = [f for f in fs if f["is_animal"] or f["is_empty"]]
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        inside = [f for f in pool if lo <= f["present"] < hi or (i == bins - 1 and f["present"] == 1.0)]
        out.append({
            "bin": f"{lo:.1f}-{hi:.1f}", "lo": lo, "hi": hi, "n": len(inside),
            "n_animal": sum(f["is_animal"] for f in inside),
            "mean_present": round(sum(f["present"] for f in inside) / len(inside), 3) if inside else None,
            "observed_animal_rate": round(sum(f["is_animal"] for f in inside) / len(inside), 3) if inside else None,
        })
    return out


def difficulty_finding(fs: list[dict]) -> dict:
    hard = [f for f in fs if f["difficulty"] == "hard"]
    night = [f["p_hard"] for f in fs if f["lighting"] == "night_ir" and f["p_hard"] is not None]
    day = [f["p_hard"] for f in fs if f["lighting"] == "day" and f["p_hard"] is not None]
    animals = [f for f in fs if f["is_animal"]]
    return {
        "n_frames": len(fs),
        "called_hard": len(hard),
        "called_hard_pct": pct(len(hard), len(fs)),
        "p_hard_auc_night_vs_day": round(auc(night, day), 4) if night and day else None,
        "animal_frames_called_hard": sum(f["difficulty"] == "hard" for f in animals),
        "n_animal": len(animals),
    }


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


def frame_costs(f: dict, tier: str) -> dict:
    w, h = f["width"], f["height"]
    lite = nova.baseline_cost("small", w, h)
    pro = nova.baseline_cost("large", w, h)
    alert_in = nova.estimate_image_tokens(w, h) + nova.estimate_text_tokens(nova.ALERT_PROMPT)
    alert = nova.cost(nova.model_for("small"), alert_in, ALERT_OUTPUT_TOKENS)
    routed = {"skip": 0.0, "small": lite, "large": pro, "escalate": alert + pro}[tier]
    gated_lite = {"skip": 0.0, "small": lite, "large": lite, "escalate": alert + lite}[tier]
    return {"lite": lite, "pro": pro, "routed": routed, "gated_lite": gated_lite}


def cost_model(fs: list[dict]) -> dict:
    scale = 10000 / len(fs)
    w, h = Counter((f["width"], f["height"]) for f in fs).most_common(1)[0][0]
    tokens = nova.estimate_image_tokens(w, h) + nova.estimate_text_tokens(nova.LABEL_TEXT)
    per = {}
    base = Counter()
    for f in fs:
        c = frame_costs(f, "skip")
        base.update({"lite": c["lite"], "pro": c["pro"]})
    for name in POLICY_NAMES:
        tot = Counter()
        for f in fs:
            tot.update(frame_costs(f, f["tiers"][name]))
        per[name] = {
            "routed": round(tot["routed"] * scale, 4),
            "lite_only_variant": round(tot["gated_lite"] * scale, 4),
            "routed_vs_lite_pct": pct(tot["routed"], base["lite"]),
            "routed_vs_pro_pct": pct(tot["routed"], base["pro"]),
            "lite_only_vs_lite_pct": pct(tot["gated_lite"], base["lite"]),
        }
    return {
        "basis": f"held-out set, {len(fs)} frames, natural empty/animal mix; scaled to 10,000 frames",
        "status": "estimated from published token formula and prices; no Bedrock call has been measured yet",
        "nova_lite_every_frame": round(base["lite"] * scale, 4),
        "nova_pro_every_frame": round(base["pro"] * scale, 4),
        "policies": per,
        "routing": "skip: $0; small: Nova Lite label; large: Nova Pro label; escalate: Nova Lite alert + Nova Pro label (pipeline default label_escalations=True)",
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


def species_eval(fs: list[dict], predict) -> dict:
    scored = [(f, predict(f)) for f in fs]
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


def nova_baselines(fs: list[dict]) -> dict | None:
    lite, pro = load_nova("lite"), load_nova("pro")
    if not lite and not pro:
        return None
    out = {}
    for name, rows in (("nova_lite", lite), ("nova_pro", pro)):
        if not rows:
            continue
        calls = list(rows.values())
        out[name] = {
            **species_eval(fs, lambda f: rows.get(f["file_name"], {}).get("species")),
            "measured_cost_per_10k_usd": round(sum(c["cost_usd"] for c in calls) / len(calls) * 10000, 4),
            "latency_ms_p50": quantile([c["latency_ms"] for c in calls], 0.5),
            "mean_input_tokens": round(sum(c["input_tokens"] for c in calls) / len(calls), 1),
        }
    if lite and pro:
        for policy in POLICY_NAMES:
            def trail(f, policy=policy):
                tier = f["tiers"][policy]
                if tier == "skip":
                    return "empty"
                return (lite if tier == "small" else pro).get(f["file_name"], {}).get("species")
            covered = [f for f in fs if trail(f) is not None]
            cost = sum((lite if f["tiers"][policy] == "small" else pro)[f["file_name"]]["cost_usd"]
                       for f in covered if f["tiers"][policy] != "skip")
            out[f"trailgate_{policy}"] = {
                **species_eval(fs, trail),
                "measured_cost_per_10k_usd": round(cost / len(covered) * 10000, 4) if covered else None,
                "note": "species = Nova Lite on small-tier frames, Nova Pro on large/escalate, 'empty' on skip; alert-call cost not included",
            }
    return out


def load_audit() -> dict:
    path = RESULTS / "label_audit.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {a["file_name"]: a for a in csv.DictReader(f)}


def audit_summary(anchors: list[dict], vehicles: list[dict], policy: str, audit: dict) -> dict:
    missed = [f for f in anchors if f["is_animal"] and f["tiers"][policy] == "skip"]
    alerts = [f for f in anchors if f["tiers"][policy] == "escalate" and not f["is_vehicle"]]
    missed_veh = [f for f in anchors + vehicles if f["is_vehicle"] and f["tiers"][policy] != "escalate"]
    seen = lambda fs: dict(Counter(audit.get(f["file_name"], {}).get("visible", "not checked") for f in fs))
    return {
        "scope": "only frames a policy got wrong by CCT labels were checked by eye; the rest of the set was not audited",
        "false_alerts_checked": len(alerts),
        "false_alerts_with_person": sum(audit.get(f["file_name"], {}).get("audited_label") == "person" for f in alerts),
        "false_alerts_not_checked": sum(f["file_name"] not in audit for f in alerts),
        "false_alert_notes": [{"file_name": f["file_name"], "note": audit[f["file_name"]]["note"]} for f in alerts
                              if f["file_name"] in audit and audit[f["file_name"]].get("audited_label") != "person"],
        "missed_animals_visible": seen(missed),
        "missed_animals_later_burst_frame": sum(int(f.get("frame_num") or 0) >= 2 for f in missed),
        "missed_vehicles": [{"file_name": f["file_name"], "escalate": f["escalate"],
                             "note": audit.get(f["file_name"], {}).get("note", "not checked")} for f in missed_veh],
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
    }


def save_gallery(missed: list[dict], audit: dict, folder: str, md_name: str, title: str, burst_max: dict) -> None:
    out = RESULTS / folder
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    lines = [f"# {title}", "",
             "\"Visible to the eye\" is a manual check of the full-size frame: yes, faint (a shape or eye shine that may be the animal), "
             "or no. Frame n/N is the frame's place in its camera burst (CCT `frame_num` / `seq_num_frames`).", "",
             "| Thumbnail | Label | Lighting | P(animal) | Max P(animal) in burst | Frame | Visible to the eye | Note | File |",
             "|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(missed, key=lambda x: x["present"]):
        img = Image.open(IMAGES / f["file_name"])
        img.thumbnail((400, 400))
        img.convert("RGB").save(out / f["file_name"], quality=85)
        a = audit.get(f["file_name"], {})
        lines.append(f"| ![{f['label']}]({folder}/{f['file_name']}) | {f['label']} | {f['lighting']} | {f['present']:.3f} | "
                     f"{burst_max[f['seq_id']]:.3f} | {f.get('frame_num', '?')}/{f.get('seq_num_frames', '?')} | "
                     f"{a.get('visible', 'not checked')} | {a.get('note', '')} | `{f['file_name']}` |")
    (RESULTS / md_name).write_text("\n".join(lines) + "\n")


def headline_rows(block: dict) -> list[str]:
    lines = ["| Subset | Frames | Empty correctly skipped | Animals missed (routed to skip) | Frames skipped | Vehicles caught | "
             "Alerts on frames without a car label | Tiers skip / small / large / escalate |",
             "|---|---|---|---|---|---|---|---|"]
    for key in ("all", *LIGHTINGS):
        s = block["heldout"][key]
        t = s["tiers"]
        lines.append(
            f"| {LIGHT_NAME[key]} | {s['n_frames']} | {s['empty_skipped']}/{s['n_empty']} ({fmt(s['empty_skipped_pct'], '%')}) | "
            f"**{s['animals_missed']}/{s['n_animal']}** ({fmt(s['animals_missed_pct'], '%')}) | {s['frames_skipped']} ({fmt(s['frames_skipped_pct'], '%')}) | "
            f"{s['vehicles_caught']}/{s['n_vehicle']} | {s['false_alerts']} | {t['skip']} / {t['small']} / {t['large']} / {t['escalate']} |")
    return lines


def sequence_rows(m: dict) -> list[str]:
    lines = ["| Policy | Subset | Bursts | Empty bursts skipped | Animal bursts missed (every frame skipped) | Animal bursts partly skipped | Vehicle bursts caught |",
             "|---|---|---|---|---|---|---|"]
    for name in POLICY_NAMES:
        for key in ("all", *LIGHTINGS):
            s = m["policies"][name]["sequences"][key]
            lines.append(f"| {name} | {LIGHT_NAME[key]} | {s['n_sequences']} | {s['empty_skipped']}/{s['n_empty']} ({fmt(s['empty_skipped_pct'], '%')}) | "
                         f"**{s['animals_missed']}/{s['n_animal']}** ({fmt(s['animals_missed_pct'], '%')}) | {s['animal_partly_skipped']} | "
                         f"{s['vehicles_caught']}/{s['n_vehicle']} |")
    return lines


def sweep_rows(m: dict) -> list[str]:
    n_empty = m["policies"]["v1"]["heldout"]["all"]["n_empty"]
    n_animal = m["policies"]["v1"]["heldout"]["all"]["n_animal"]
    lines = ["| skip_below | v1: frames skipped | v1: empty skipped | v1: animals missed | v2: frames skipped | v2: empty skipped | v2: animals missed |",
             "|---|---|---|---|---|---|---|"]
    marks = {m["thresholds"]["v1"]["skip_below"], m["thresholds"]["skip_below"]}
    for a, b in zip(m["policies"]["v1"]["sweep"], m["policies"]["v2"]["sweep"]):
        s = a["skip_below"]
        if round(s * 100) % 5 == 0 or s in marks:
            row = (f"| {s:.2f} | {a['frames_skipped']} ({a['frames_skipped_pct']}%) | {a['empty_skipped']}/{n_empty} | {a['animals_missed']}/{n_animal} | "
                   f"{b['frames_skipped']} ({b['frames_skipped_pct']}%) | {b['empty_skipped']}/{n_empty} | {b['animals_missed']}/{n_animal} |")
            if s in marks:
                row = "| " + " | ".join(f"**{c.strip()}**" for c in row.strip("|").split("|")) + " |"
            lines.append(row)
    return lines


def audit_lines(a: dict, n_missed: int) -> list[str]:
    v = a["missed_animals_visible"]
    return [
        f"- Alerts on frames without a `car` label: {a['false_alerts_checked']}; {a['false_alerts_with_person']} show people "
        f"(CCT labels people `empty`, so these alerts are correct), "
        f"{a['false_alerts_checked'] - a['false_alerts_with_person'] - a['false_alerts_not_checked']} checked by eye with no person or vehicle "
        f"(true false alerts), {a['false_alerts_not_checked']} not checked."
        + ("".join(f" `{x['file_name'][:8]}`: {x['note']}." for x in a["false_alert_notes"]) if a.get("false_alert_notes") else ""),
        f"- Missed animal frames ({n_missed}): visible to the eye {v.get('yes', 0)}, faint {v.get('faint', 0)}, nothing visible {v.get('no', 0)}, "
        f"not checked {v.get('not checked', 0)}. {a['missed_animals_later_burst_frame']} are frame 2 or later of their burst.",
        f"- Missed vehicle frames ({len(a['missed_vehicles'])}): "
        + ("; ".join(f"`{x['file_name'][:8]}` P={x['escalate']:.2f}, {x['note']}" for x in a["missed_vehicles"]) or "none") + ".",
    ]


def write_markdown(m: dict) -> str:
    t1, t2 = m["thresholds"]["v1"], m["thresholds"]
    v1, v2f, v2 = (m["policies"][k] for k in ("v1", "v2_frame", "v2"))
    h = v1["heldout"]["all"]
    c = m["cost"]
    d = m["difficulty"]
    b = m["bursts"]
    lines = [
        "# Trailgate evaluation",
        "",
        f"Held-out set: {h['n_frames']} Caltech Camera Traps frames from camera locations never used for tuning, natural mix "
        f"({h['n_empty']} empty, {h['n_animal']} animal, {h['n_vehicle']} car), in {b['n_sequences']} camera bursts with "
        f"{b['n_burst_frames']} frames in all. Vehicle set: {v1['vehicle_set']['n_frames']} `car` frames from held-out locations. "
        "One Decider call per frame, the same 3-question production call for every frame. "
        "Thresholds were set on the tuning set only and frozen before the held-out run. Small set; 0 means no errors on this set, not solved.",
        "",
    ]
    if m.get("dataset"):
        ds = m["dataset"]
        lines += [f"Full CCT label file (`{ds['source']}`): {ds['n_empty']:,} of {ds['n_images']:,} images are empty ({ds['empty_pct']}%; "
                  "lila.science's summary says about 70%). The label set has no person class; CCT labels whole bursts.", ""]
    lines += [
        "## Summary: v1 to v2",
        "",
        "| Policy | Empty frames skipped | Animal frames missed | Empty bursts skipped | Animal bursts missed | Vehicles caught | Tiers small / large | $ per 10,000 frames |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in POLICY_NAMES:
        p = m["policies"][name]
        s, q = p["heldout"]["all"], p["sequences"]["all"]
        lines.append(f"| {POLICY_NAMES[name]} | {s['empty_skipped']}/{s['n_empty']} ({s['empty_skipped_pct']}%) | "
                     f"{s['animals_missed']}/{s['n_animal']} ({s['animals_missed_pct']}%) | {q['empty_skipped']}/{q['n_empty']} ({q['empty_skipped_pct']}%) | "
                     f"{q['animals_missed']}/{q['n_animal']} ({q['animals_missed_pct']}%) | {p['vehicles']['caught']}/{p['vehicles']['n']} | "
                     f"{s['tiers']['small']} / {s['tiers']['large']} | ${c['policies'][name]['routed']:.2f} |")
    lines += [
        f"| Nova Lite on every frame | 0 | 0 | 0 | 0 | n/a | {h['n_frames']} / 0 | ${c['nova_lite_every_frame']:.2f} |",
        f"| Nova Pro on every frame | 0 | 0 | 0 | 0 | n/a | 0 / {h['n_frames']} | ${c['nova_pro_every_frame']:.2f} |",
        "",
        "Costs are estimated from the published token formula and prices; no Bedrock call has been measured yet.",
        "",
        "## v1 (frozen, as committed)",
        "",
        f"Thresholds: skip below {t1['skip_below']}, small at or above {t1['small_above']} only when Decider calls the subject clear, "
        f"escalate at or above {t1['escalate_above']}. Rule: {t1['method']}",
        "",
        *headline_rows(v1),
        "",
        f"Tuning set vs held-out: presence AUC {m['tune_reference']['auc']} on tuning vs {h['presence_auc_animal_vs_empty']} held-out; "
        f"{m['tune_reference']['neg_below_zero_miss'] * 100:.1f}% of tuning empty frames below skip_below with 0 animals below, vs "
        f"{h['empty_skipped_pct']}% held-out with {h['animals_missed']} animals below. The threshold did not transfer without misses."
        if m.get("tune_reference") else "",
        "",
        *audit_lines(v1["audit"], v1["heldout"]["all"]["animals_missed"]),
        "",
        "## Finding: the difficulty question carries little signal",
        "",
        f"On the held-out frames Decider called {d['called_hard']} of {d['n_frames']} frames \"hard\" ({d['called_hard_pct']}%), including "
        f"{d['animal_frames_called_hard']} of {d['n_animal']} animal frames. P(hard) separates night infrared from day frames with AUC "
        f"{d['p_hard_auc_night_vs_day']}. Because v1 sent every \"hard\" frame to Nova Pro, {v1['heldout']['all']['tiers']['large']} of 600 "
        f"frames went to the large model. v2 still asks the question (it is nearly free in the batched call) but does not route on it.",
        "",
        "## v2: confidence band + burst context",
        "",
        f"Policy: escalate if P(person/vehicle) >= {t2['escalate_above']} in any frame of the burst; skip a frame only if max P(animal) over its "
        f"burst < {t2['skip_below']}; Nova Lite if the frame's P(animal) >= {t2['small_above']}; otherwise Nova Pro (Decider is unsure). "
        f"Rule: {t2['method']}",
        "",
        *headline_rows(v2),
        "",
        *audit_lines(v2["audit"], v2["heldout"]["all"]["animals_missed"]),
        "",
        "### Ablation: v2 thresholds without burst context",
        "",
        *headline_rows(v2f),
        "",
        "## Sequence level (CCT's labelling unit: one label per burst)",
        "",
        f"All {b['n_burst_frames']} frames of the {b['n_sequences']} held-out bursts were routed. A burst counts as skipped only when every one of its frames is skipped. "
        "Burst labels are the held-out frame's CCT label.",
        "",
        *sequence_rows(m),
        "",
    ]
    mi = m.get("multi_image")
    if mi:
        lines += [
            "### One multi-image call per burst vs max over frames (tuning bursts)",
            "",
            f"{mi['n_bursts']} tuning bursts with 2-4 frames, the presence question asked once with all frames attached. "
            f"AUC animal vs empty burst: multi-image {mi['auc_multi']} vs max-of-frames {mi['auc_max']}. "
            f"Mean |difference| {mi['mean_abs_diff']}; same side of skip_below={t2['skip_below']} on {mi['agree_at_skip']}/{mi['n_bursts']} bursts; "
            f"animal bursts below skip_below: multi-image {mi['animal_below_multi']}, max-of-frames {mi['animal_below_max']}. "
            f"p50 latency per burst: one multi-image call {mi['latency_multi_p50']} ms vs one call per frame {mi['latency_frames_p50']} ms. "
            f"Adopted: {mi['adopted']}.",
            "",
        ]
    lines += [
        "## The safety knob: skip threshold against animals missed (held-out frames)",
        "",
        "Other thresholds held at each policy's frozen values. Frozen points in bold.",
        "",
        *sweep_rows(m),
        "",
        "## Reliability (held-out, animal vs empty frames)",
        "",
        "| P(animal) bin | Frames | Animal frames | Mean P(animal) | Observed animal rate |",
        "|---|---|---|---|---|",
        *[f"| {r['bin']} | {r['n']} | {r['n_animal']} | {fmt(r['mean_present'])} | {fmt(r['observed_animal_rate'])} |" for r in m["reliability"]],
        "",
        "## Cost per 10,000 frames",
        "",
        f"**{c['status']}.** Basis: {c['basis']}. Routing cost: {c['routing']}.",
        "",
        "| System | $ per 10,000 frames | vs Nova Lite everywhere | vs Nova Pro everywhere |",
        "|---|---|---|---|",
        f"| Nova Lite on every frame | ${c['nova_lite_every_frame']:.2f} | 100% | {pct(c['nova_lite_every_frame'], c['nova_pro_every_frame'])}% |",
        f"| Nova Pro on every frame | ${c['nova_pro_every_frame']:.2f} | {pct(c['nova_pro_every_frame'], c['nova_lite_every_frame'])}% | 100% |",
    ]
    for name in POLICY_NAMES:
        p = c["policies"][name]
        lines.append(f"| Trailgate {POLICY_NAMES[name]} | ${p['routed']:.2f} | {p['routed_vs_lite_pct']}% | {p['routed_vs_pro_pct']}% |")
    for name in ("v1", "v2"):
        p = c["policies"][name]
        lines.append(f"| Trailgate {name}, Lite-only variant (skip, else Nova Lite) | ${p['lite_only_variant']:.2f} | {p['lite_only_vs_lite_pct']}% | "
                     f"{pct(p['lite_only_variant'], c['nova_pro_every_frame'])}% |")
    tok = c["label_call_tokens_typical_frame"]
    lines += [
        "",
        f"Per label call on a {tok['frame']} frame: ~{tok['input']} input tokens (image estimate from the closest documented resolution "
        f"plus the prompt) and {tok['output']} output tokens. On-demand prices per 1K tokens (us-east-1): "
        + "; ".join(f"{k} ${v['input']} in / ${v['output']} out" for k, v in c["prices_per_1k_tokens_usd"].items()) + ".",
        "",
        f"Decider runs locally: $0 per call, on {c['decider_machine']}. p50 / p90 Decider latency per frame: "
        f"{m['decider_latency_ms']['p50']} / {m['decider_latency_ms']['p90']} ms over {m['decider_latency_ms']['n']} calls. "
        "The burst rule needs no extra calls in deployment, because every frame of a burst is asked anyway; it waits for the burst to finish.",
        "",
        "Sources:",
        "",
        *[f"- {s}" for s in c["sources"]],
    ]
    if m.get("nova"):
        lines += ["", "## Measured Bedrock baselines (held-out)", "",
                  "| System | Frames scored | Species accuracy on animal frames | Empty frames called empty | Animals called empty | Measured $ per 10,000 |",
                  "|---|---|---|---|---|---|"]
        for name, s in m["nova"].items():
            lines.append(f"| {name} | {s['n_scored']} | {fmt(s['species_accuracy_on_animals'], '%')} | {s['empty_correct']} "
                         f"({fmt(s['empty_correct_pct'], '%')}) | {s['animals_said_empty']} | {fmt(s['measured_cost_per_10k_usd'])} |")
    else:
        lines += ["", "Bedrock baselines: not yet measured (Bedrock on-demand quota on this account was 0 at evaluation time; "
                  "`scripts/run_nova_baseline.py` fills `results/nova_{lite,pro}.jsonl` and this script merges them)."]
    bc = m.get("batching_check")
    if bc:
        lines += ["", f"Batching check: on {bc['n_frames']} tuning frames, asking the same wording in a 9-question call and in the 3-question "
                  f"production call changed P(animal) by at most {bc['present_max_abs_diff']} and P(person/vehicle) by at most "
                  f"{bc['escalate_max_abs_diff']}; difficulty flipped on {bc['difficulty_flips']} frames."]
    lines += ["", f"Missed-animal galleries: `results/missed.md` (v2, {v2['heldout']['all']['animals_missed']} frames) and "
              f"`results/missed_v1.md` (v1, {v1['heldout']['all']['animals_missed']} frames).", ""]
    return "\n".join(lines)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-ask", action="store_true")
    ap.add_argument("--ask-only", action="store_true")
    ap.add_argument("--cache", default=str(RESULTS / "decider_answers.jsonl"))
    args = ap.parse_args(argv)

    questions = build_questions()
    heldout = read_manifest(ROOT / "data/manifests/heldout.csv")
    bursts = read_manifest(ROOT / "data/manifests/heldout_bursts.csv")
    vehicle = read_manifest(ROOT / "data/manifests/vehicle.csv")
    if args.no_ask:
        cache = load_cache(Path(args.cache), question_signature(questions, STATE))
    else:
        cache = ask_all(heldout + bursts + vehicle, questions, Path(args.cache), IMAGES)
    if args.ask_only:
        return {}
    tfile = json.loads((RESULTS / "thresholds.json").read_text())
    policies = {"v1": (thresholds_of(tfile["v1"] | {"use_difficulty": True}), False),
                "v2_frame": (thresholds_of(tfile), False),
                "v2": (thresholds_of(tfile), True)}

    anchor_label = {r["file_name"]: r for r in heldout}
    for b in bursts:
        if b["file_name"] in anchor_label:
            b["label"] = anchor_label[b["file_name"]]["label"]
            b["lighting"] = anchor_label[b["file_name"]]["lighting"]
    bf = frame_rows(bursts, cache, "heldout")
    vf = frame_rows(vehicle, cache, "vehicle")
    seq_label = {}
    for f in bf:
        if f["anchor"]:
            seq_label[f["seq_id"]] = f["label"]
    for f in bf:
        f["seq_label"] = seq_label[f["seq_id"]]
    for name, (t, burst) in policies.items():
        for f, tier in zip(bf, route_all(bf, t, burst)):
            f.setdefault("tiers", {})[name] = tier
        for f, tier in zip(vf, route_all(vf, t, burst)):
            f.setdefault("tiers", {})[name] = tier
    anchors = [f for f in bf if f["anchor"]]
    assert len(anchors) == len(heldout)
    burst_max = defaultdict(float)
    for f in bf:
        burst_max[f["seq_id"]] = max(burst_max[f["seq_id"]], f["present"])

    audit = load_audit()
    out_policies = {}
    for name, (t, burst) in policies.items():
        for f in anchors + vf + bf:
            f["tier"] = f["tiers"][name]
        hs = {"all": summary(anchors), **{l: summary([f for f in anchors if f["lighting"] == l]) for l in LIGHTINGS}}
        vs = summary(vf)
        caught = hs["all"]["vehicles_caught"] + vs["vehicles_caught"]
        n_veh = hs["all"]["n_vehicle"] + vs["n_vehicle"]
        out_policies[name] = {
            "description": POLICY_NAMES[name],
            "thresholds": {"skip_below": t.skip_below, "small_above": t.small_above, "escalate_above": t.escalate_above,
                           "use_difficulty": t.use_difficulty, "burst_aware": burst},
            "heldout": hs,
            "vehicle_set": vs,
            "vehicles": {"caught": caught, "n": n_veh, "caught_pct": pct(caught, n_veh)},
            "sequences": sequence_summary(bf, seq_label),
            "sweep": sweep(bf, t, burst),
            "audit": audit_summary(anchors, vf, name, audit),
            "missed": [{k: f.get(k) for k in ("file_name", "label", "lighting", "location", "present", "frame_num", "seq_num_frames")}
                       | {"burst_max_present": burst_max[f["seq_id"]]} for f in anchors if f["is_animal"] and f["tier"] == "skip"],
        }
        if name == "v1":
            out_policies[name]["label_noise_candidates"] = [
                {k: f[k] for k in ("file_name", "lighting", "present", "escalate")} for f in anchors
                if f["is_empty"] and f["escalate"] >= t.escalate_above]
    lat = [f["latency_ms"] for f in bf + vf]
    metrics = {
        "thresholds": tfile,
        "questions": questions,
        "state": STATE,
        "bursts": {"n_sequences": len(seq_label), "n_burst_frames": len(bf), "n_anchor_frames": len(anchors),
                   "burst_sizes": dict(Counter(int(f["seq_num_frames"]) for f in anchors))},
        "policies": out_policies,
        "difficulty": difficulty_finding(anchors),
        "decider_latency_ms": {"n": len(lat), "p50": quantile(lat, 0.5), "p90": quantile(lat, 0.9),
                               "note": "wall clock around one HTTP call with the 3 production questions, all held-out burst and vehicle frames"},
        "reliability": reliability(anchors),
        "reliability_by_lighting": {l: reliability([f for f in anchors if f["lighting"] == l]) for l in LIGHTINGS},
        "cost": cost_model(anchors),
        "nova": nova_baselines(anchors),
        "dataset": dataset_stats(),
        "tune_reference": next((r for r in json.loads((RESULTS / "wording.json").read_text())["presence"]
                                if r["wording"] == tfile["chosen_wording"]), None) if (RESULTS / "wording.json").exists() else None,
        "batching_check": batching_check(),
        "multi_image": json.loads((RESULTS / "burst_multi_image.json").read_text()) if (RESULTS / "burst_multi_image.json").exists() else None,
        "code_check_skips": [{k: f[k] for k in ("file_name", "label", "check")} for f in bf + vf if f["check"]],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS / "metrics.md").write_text(write_markdown(metrics))
    with open(RESULTS / "eval_frames.jsonl", "w") as fh:
        for f in bf + vf:
            fh.write(json.dumps({k: v for k, v in f.items() if k not in ("answers", "tier")}) + "\n")
    for name, folder, md, title in (("v2", "missed", "missed.md", "Animal frames Trailgate v2 skipped (held-out)"),
                                    ("v1", "missed_v1", "missed_v1.md", "Animal frames Trailgate v1 skipped (held-out)")):
        save_gallery([f for f in anchors if f["is_animal"] and f["tiers"][name] == "skip"], audit, folder, md, title, burst_max)
    from charts import draw_all
    draw_all(metrics)
    print(write_markdown(metrics))
    return metrics


if __name__ == "__main__":
    main()
