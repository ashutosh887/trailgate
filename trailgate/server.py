"""Trailgate demo server: a night's camera-trap card, triaged frame by frame over SSE.

Modes:
  live    each frame goes through trailgate.pipeline.Triage (Decider at 127.0.0.1:8099, then Bedrock)
  replay  results are read from a cached jsonl in results/ and paced like a live run

Run:
  python -m trailgate.server   (replays results/decider_answers.jsonl with the v2 burst policy)
  python -m trailgate.server --replay results/decider_answers.jsonl
  python -m trailgate.server --mode live --record results/run-live.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import logging
import os
import random
import socket
import statistics
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from patterns.presence_router import PresenceRouterCore, Thresholds, build_questions
from patterns.presence_router.router import DEFAULT_URL
from trailgate.results import FrameResult, ModelCall

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent / "web"
IMAGES = ROOT / "data" / "images"
MANIFESTS = ROOT / "data"
RESULTS = ROOT / "results"
OUTBOX = Path(os.environ.get("TRAILGATE_OUTBOX", ROOT / "outbox"))

TIER_ACTION = {
    "skip": "EMPTY",
    "small": "LABEL",
    "large": "LABEL",
    "escalate": "ALERT",
}
TIER_MODEL = {"small": "Nova Lite", "large": "Nova Pro", "escalate": "Nova Lite"}

FALLBACK_PRICING = {"Nova Lite": (0.00006, 0.00024), "Nova Pro": (0.0008, 0.0032)}
FALLBACK_TOKENS_PER_FRAME = {"in": 1290, "out": 15}

REELS = [
    {"id": "card-a", "name": "Card A", "seed": 7, "size": 54},
    {"id": "card-b", "name": "Card B", "seed": 23, "size": 54},
    {"id": "card-c", "name": "Card C", "seed": 41, "size": 54},
]
DEFAULT_REPLAY = RESULTS / "decider_answers.jsonl"
LABEL_AUDIT = RESULTS / "label_audit.csv"
DIFFICULTY_NOTE = "asked, not used for routing · weak signal: AUC 0.67 night vs day"

CONFIG = {
    "mode": "replay",
    "replay": None,
    "record": None,
    "pace": 1.0,
    "decider_url": DEFAULT_URL,
}

app = FastAPI(title="Trailgate")
log = logging.getLogger("trailgate.server")
_run_lock = asyncio.Lock()
_write_lock = threading.Lock()


def _read_manifest(name: str) -> list[dict]:
    path = MANIFESTS / f"{name}.csv"
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        rows = [{k.strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(fh)]
    for row in rows:
        row["source"] = name
    return rows


@lru_cache(maxsize=1)
def manifest() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for name in ("tune", "vehicle", "heldout_bursts", "heldout"):
        for row in _read_manifest(name):
            rows[row["file_name"]] = {**rows.get(row["file_name"], {}), **row}
    return rows


@lru_cache(maxsize=1)
def bursts() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in _read_manifest("heldout_bursts"):
        out.setdefault(row["seq_id"], []).append(row)
    for frames in out.values():
        frames.sort(key=lambda r: int(r.get("frame_num") or 0))
    return out


def burst_of(file_name: str) -> list[dict]:
    """The frames of the camera burst this frame belongs to; a lone frame is its own burst."""
    row = manifest().get(file_name, {})
    frames = bursts().get(row.get("seq_id", ""), [])
    return frames if any(f["file_name"] == file_name for f in frames) else [row]


@lru_cache(maxsize=1)
def audited_people() -> list[str]:
    """Held-out frames CCT labels empty that show people, from the label audit."""
    if not LABEL_AUDIT.exists():
        return []
    with LABEL_AUDIT.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return sorted(r["file_name"] for r in rows if r.get("set") == "heldout" and r.get("audited_label") == "person")


@lru_cache(maxsize=1)
def _pricing() -> dict:
    """Per-frame label-call estimates: from trailgate.bedrock when importable, else FALLBACK_PRICING."""
    try:
        from trailgate import bedrock

        tokens_in = bedrock.estimate_image_tokens(1024, 747) + bedrock.estimate_text_tokens(bedrock.LABEL_TEXT)
        return {
            "Nova Lite": bedrock.baseline_cost("small"),
            "Nova Pro": bedrock.baseline_cost("large"),
            "tokens": tokens_in + bedrock.LABEL_OUTPUT_TOKENS,
            "source": "trailgate.bedrock.baseline_cost",
        }
    except (ImportError, AttributeError):
        t = FALLBACK_TOKENS_PER_FRAME
        per = {k: (t["in"] * v[0] + t["out"] * v[1]) / 1000 for k, v in FALLBACK_PRICING.items()}
        return {**per, "tokens": t["in"] + t["out"], "source": "FALLBACK_PRICING (server.py)"}


def _model_name(model_id: str, tier: str) -> str:
    mid = (model_id or "").lower()
    if "pro" in mid:
        return "Nova Pro"
    if "lite" in mid:
        return "Nova Lite"
    return TIER_MODEL.get(tier, "Nova Lite")


def costing(result: dict) -> dict:
    """What the frame cost, what it would have cost on Nova Pro, and whether any part is an estimate."""
    p = _pricing()
    actual, est, tok, tok_est = 0.0, 0.0, 0, 0
    calls = result.get("calls") or []
    for call in calls:
        if call.get("error"):
            est += p[_model_name(call.get("model_id", ""), call.get("tier", result["tier"]))]
            tok_est += p["tokens"]
        else:
            actual += float(call.get("cost_usd") or 0.0)
            tok += int(call.get("input_tokens") or 0) + int(call.get("output_tokens") or 0)
    if not calls and result["tier"] in TIER_MODEL:
        est += p[TIER_MODEL[result["tier"]]]
        tok_est += p["tokens"]
        if result["tier"] == "escalate":
            est += p["Nova Lite"]
            tok_est += p["tokens"]
    return {
        "actual_usd": actual,
        "estimated_usd": est,
        "tokens": tok,
        "tokens_estimated": tok_est,
        "baseline_usd": p["Nova Pro"],
        "baseline_lite_usd": p["Nova Lite"],
        "baseline_tokens": p["tokens"],
        "pricing_source": p["source"],
    }


def _frame_key(value: str | None) -> str | None:
    if not value:
        return None
    name = Path(str(value)).name
    return name if name.endswith(".jpg") else f"{name}.jpg"


PRESENT_KEYS = ("present", "p_criteria", "animal_present", "p_plain")
ESCALATE_KEYS = ("escalate", "e_criteria", "human_present", "e_plain")


def _pick(answers: dict, keys: tuple[str, ...], prefix: str) -> dict | None:
    for key in keys:
        if isinstance(answers.get(key), dict) and "noul" in answers[key]:
            return answers[key]
    for key, value in answers.items():
        if key.startswith(prefix) and isinstance(value, dict) and "noul" in value:
            return value
    return None


def _answers_of(row: dict) -> dict | None:
    """Decider answers in router shape {present, escalate, difficulty}, mapping eval-style keys when needed."""
    nested = [(row.get(k) or {}).get("answers") for k in ("response", "decider")]
    for candidate in (row.get("answers"), *nested):
        if not isinstance(candidate, dict):
            continue
        present = _pick(candidate, PRESENT_KEYS, "p_")
        difficulty = candidate.get("difficulty")
        if present is None or not isinstance(difficulty, dict):
            continue
        out = {"present": present, "difficulty": difficulty}
        escalate = _pick(candidate, ESCALATE_KEYS, "e_")
        if escalate is not None:
            out["escalate"] = escalate
        return out
    return None


def _normalise(row: dict) -> tuple[str, dict] | None:
    """Accept a recorded FrameResult row or a raw Decider-answers row; return (file_name, FrameResult dict)."""
    key = _frame_key(row.get("frame") or row.get("file_name") or row.get("image") or row.get("path") or row.get("id"))
    if not key:
        return None
    if "tier" in row and "reason" in row and "present" in row:
        result = dict(row)
        result.setdefault("calls", [])
        result["frame"] = key
        result["kind"] = "recorded"
        return key, result
    answers = _answers_of(row)
    if answers is None:
        return None
    latency = float(
        row.get("latency_ms")
        or row.get("decider_latency_ms")
        or row.get("server_latency_ms")
        or (row.get("response") or {}).get("latency_ms")
        or 0.0
    )
    route = _core().decide(answers, latency)
    meta = manifest().get(key, {})
    result = FrameResult(
        frame=key,
        tier=route.tier.value,
        present=route.present,
        escalate=route.escalate,
        difficulty=route.difficulty,
        reason=route.reason,
        decider_latency_ms=route.latency_ms,
        label=row.get("label") or meta.get("label"),
        lighting=row.get("lighting") or meta.get("lighting"),
    ).to_dict()
    result["answers"] = answers
    result["difficulty_confidence"] = route.difficulty_confidence
    result["kind"] = "answers"
    return key, result


def thresholds() -> Thresholds:
    try:
        from trailgate.pipeline import load_thresholds

        return load_thresholds()
    except (ImportError, OSError, ValueError, TypeError):
        return Thresholds()


@lru_cache(maxsize=1)
def _core() -> PresenceRouterCore:
    return PresenceRouterCore(thresholds=thresholds())


def replay_sources() -> list[Path]:
    if not RESULTS.exists():
        return []
    return sorted(RESULTS.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def replay_path() -> Path | None:
    if CONFIG["replay"]:
        return Path(CONFIG["replay"])
    if DEFAULT_REPLAY.exists():
        return DEFAULT_REPLAY
    sources = replay_sources()
    if not sources:
        return None
    return max(sources, key=lambda p: (len(load_replay(p)), p.stat().st_mtime))


def load_replay(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    stat = path.stat()
    return _load_replay_cached(str(path), stat.st_mtime, stat.st_size)


@lru_cache(maxsize=8)
def _load_replay_cached(path: str, mtime: float, size: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = _normalise(json.loads(line))
            except (ValueError, KeyError, TypeError):
                continue
            if parsed and (IMAGES / parsed[0]).exists():
                out[parsed[0]] = parsed[1]
    return out


def _night_order(row: dict) -> tuple:
    try:
        ts = datetime.strptime(row["captured"], "%Y-%m-%d %H:%M:%S")
    except (KeyError, ValueError):
        return (99, row["file_name"])
    return ((ts.hour - 18) % 24, ts.minute, ts.second, row["file_name"])


def replay_result(file_name: str, cache: dict[str, dict]) -> dict:
    """The cached result for a frame; raw Decider answers are re-routed with the burst rule over its siblings."""
    own = cache[file_name]
    if own.get("kind") != "answers":
        return json.loads(json.dumps(own))
    core = _core()
    siblings = [f["file_name"] for f in burst_of(file_name) if f.get("file_name") in cache]
    if file_name not in siblings:
        siblings = [file_name]
    routes = [core.decide(cache[f]["answers"], float(cache[f]["decider_latency_ms"] or 0.0)) for f in siblings]
    decided = core.decide_burst(routes) if len(routes) > 1 else routes
    route = decided[siblings.index(file_name)]
    result = json.loads(json.dumps(own))
    result.update(tier=route.tier.value, reason=route.reason)
    result["burst"] = [
        {"file_name": f, "present": r.present, "escalate": r.escalate, "self": f == file_name, "thumb": f"/thumb/{f}"}
        for f, r in zip(siblings, routes, strict=False)
    ]
    return result


def _shuffled(rows: list[dict], rng: random.Random) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["file_name"])
    rng.shuffle(rows)
    return rows


def build_reel(seed: int, size: int, cache: dict[str, dict] | None = None) -> list[dict]:
    """Deterministic card from held-out and vehicle frames, with at least one vehicle alert and one audited person."""
    rng = random.Random(seed)
    rows = [r for r in manifest().values() if (IMAGES / r["file_name"]).exists()]
    if cache is not None:
        rows = [r for r in rows if r["file_name"] in cache]

    def tier(row: dict) -> str | None:
        return replay_result(row["file_name"], cache)["tier"] if cache is not None else None

    people_names = audited_people()
    people = [r for r in rows if r["file_name"] in people_names and r["source"] == "heldout"]
    alerting = [r for r in people if tier(r) in (None, "escalate")] or people
    alerting.sort(key=lambda r: r["file_name"])
    pick = [alerting[seed % len(alerting)]] if alerting else []

    vehicles = [r for r in rows if r["source"] == "vehicle"]
    caught = [r for r in vehicles if tier(r) in (None, "escalate")] or vehicles
    night = _shuffled([r for r in caught if r["lighting"] == "night_ir"], rng)
    day = _shuffled([r for r in caught if r["lighting"] != "night_ir"], rng)
    pick += (night[:1] + day)[:2]

    held = [r for r in rows if r["source"] == "heldout" and r["file_name"] not in people_names]
    empties = _shuffled([r for r in held if r["label"] == "empty"], rng)
    animals = _shuffled([r for r in held if r["label"] not in ("empty", "car")], rng)
    n_rest = size - len(pick)
    n_empty = min(len(empties), round(n_rest * 0.55))
    n_animal = min(len(animals), n_rest - n_empty)
    reel = pick + empties[:n_empty] + animals[:n_animal]
    reel.sort(key=_night_order)
    return [_frame_meta(r, i) for i, r in enumerate(reel)]


def _frame_meta(row: dict, index: int) -> dict:
    captured = row.get("captured", "")
    clock = captured[11:16] if len(captured) >= 16 else ""
    return {
        "index": index,
        "file_name": row["file_name"],
        "id": row["id"],
        "label": row["label"],
        "location": row["location"],
        "captured": captured,
        "clock": clock,
        "lighting": row["lighting"],
        "burst": [f["file_name"] for f in burst_of(row["file_name"]) if f.get("file_name")],
        "audit": "person" if row["file_name"] in audited_people() else None,
        "thumb": f"/thumb/{row['file_name']}",
        "full": f"/frames/{row['file_name']}",
    }


def decider_reachable(url: str) -> bool:
    """TCP connect only: never sends a request to the Decider server."""
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=0.4):
            return True
    except OSError:
        return False


def pipeline_status() -> dict:
    try:
        from trailgate import pipeline
    except ImportError as exc:
        detail = f"trailgate.pipeline not importable ({type(exc).__name__}); router-only adapter"
        return {"available": False, "detail": detail}
    return {"available": hasattr(pipeline, "Triage"), "detail": "trailgate.pipeline.Triage"}


class _RouterOnlyTriage:
    """Stand-in for trailgate.pipeline.Triage: Decider decides, Bedrock is reported as queued."""

    def __init__(self, bedrock: bool = True):
        self.router = PresenceRouterCore(thresholds=thresholds())

    def run(self, path, label=None, lighting=None) -> FrameResult:
        route = self.router.route(path)
        calls = []
        if route.tier.value in TIER_MODEL:
            calls.append(
                ModelCall(
                    model_id=TIER_MODEL[route.tier.value],
                    tier=route.tier.value,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0.0,
                    cost_usd=0.0,
                    error="pipeline not available",
                )
            )
        return FrameResult(
            frame=str(path),
            tier=route.tier.value,
            present=route.present,
            escalate=route.escalate,
            difficulty=route.difficulty,
            reason=route.reason,
            decider_latency_ms=route.latency_ms,
            calls=calls,
            label=label,
            lighting=lighting,
        )


class LiveTriage:
    """Wraps the pipeline: runs the frame's whole burst and keeps the Decider answers so the UI can show them."""

    def __init__(self):
        try:
            from trailgate.pipeline import Triage
        except ImportError:
            Triage = _RouterOnlyTriage
        self.inner = Triage(bedrock=True)
        self.routes: list = []
        original = self.inner.router.route

        def capture(image):
            route = original(image)
            self.routes.append(route)
            return route

        self.inner.router.route = capture

    def run(self, path: Path, label: str | None, lighting: str | None) -> dict:
        self.routes = []
        names = [f["file_name"] for f in burst_of(path.name) if f.get("file_name")] or [path.name]
        paths = [IMAGES / n for n in names if (IMAGES / n).exists()]
        if path not in paths or not hasattr(self.inner, "run_burst"):
            paths = [path]
        metas = [manifest().get(p.name, {}) for p in paths]
        if len(paths) > 1:
            results = self.inner.run_burst(paths, [m.get("label") for m in metas], [m.get("lighting") for m in metas])
        else:
            results = [self.inner.run(path, label=label, lighting=lighting)]
        at = paths.index(path)
        data = results[at].to_dict() if hasattr(results[at], "to_dict") else asdict(results[at])
        data["frame"] = _frame_key(data.get("frame")) or data.get("frame")
        if len(self.routes) == len(paths):
            data["answers"] = self.routes[at].answers
            data["difficulty_confidence"] = self.routes[at].difficulty_confidence
            if len(paths) > 1:
                data["burst"] = [
                    {
                        "file_name": p.name,
                        "present": r.present,
                        "escalate": r.escalate,
                        "self": p == path,
                        "thumb": f"/thumb/{p.name}",
                    }
                    for p, r in zip(paths, self.routes, strict=False)
                ]
        return data


_triage: LiveTriage | None = None


def triage() -> LiveTriage:
    global _triage
    if _triage is None:
        _triage = LiveTriage()
    return _triage


def _record(row: dict) -> None:
    if not CONFIG["record"]:
        return
    path = Path(CONFIG["record"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock, path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def _enrich(result: dict, meta: dict, source: str) -> dict:
    result.setdefault("difficulty_confidence", (result.get("answers") or {}).get("difficulty", {}).get("confidence"))
    for call in result.get("calls") or []:
        call["model"] = _model_name(call.get("model_id", ""), call.get("tier", result["tier"]))
    return {
        "index": meta["index"],
        "meta": meta,
        "result": result,
        "action": TIER_ACTION.get(result["tier"], result["tier"].upper()),
        "model": TIER_MODEL.get(result["tier"]),
        "costing": costing(result),
        "source": source,
    }


def _replay_delay(result: dict, pace: float) -> float:
    base = {"skip": 0.32, "small": 0.85, "large": 1.15, "escalate": 1.4}.get(result["tier"], 0.6)
    jitter = min(0.25, float(result.get("decider_latency_ms") or 0) / 4000)
    return max(0.05, (base + jitter) * pace)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _digest(rows: list[dict]) -> dict:
    tiers = [r["result"]["tier"] for r in rows]
    species: dict[str, int] = {}
    for r in rows:
        s = r["result"].get("species")
        if s:
            species[s] = species.get(s, 0) + 1
    lat = [float(r["result"].get("decider_latency_ms") or 0) for r in rows if r["result"].get("decider_latency_ms")]
    spent = sum(r["costing"]["actual_usd"] + r["costing"]["estimated_usd"] for r in rows)
    baseline = sum(r["costing"]["baseline_usd"] for r in rows)
    baseline_lite = sum(r["costing"]["baseline_lite_usd"] for r in rows)
    return {
        "frames": len(rows),
        "empty": tiers.count("skip"),
        "lite": tiers.count("small"),
        "pro": tiers.count("large"),
        "labelled": tiers.count("small") + tiers.count("large"),
        "alerts": tiers.count("escalate"),
        "species": species,
        "decider_p50_ms": statistics.median(lat) if lat else None,
        "spent_usd": spent,
        "baseline_usd": baseline,
        "baseline_lite_usd": baseline_lite,
        "alert_frames": [r["meta"] for r in rows if r["result"]["tier"] == "escalate"],
    }


def _resolve_reel(reel_id: str, mode: str) -> tuple[list[dict], dict[str, dict], str]:
    reel = next((r for r in REELS if r["id"] == reel_id), REELS[0])
    if mode == "replay":
        path = replay_path()
        cache = load_replay(path)
        frames = build_reel(reel["seed"], reel["size"], cache) if cache else []
        return frames, cache, _display(path) or ""
    return build_reel(reel["seed"], reel["size"]), {}, "live"


def _display(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/status")
def status() -> dict:
    return status_info()


def status_info(probe_decider: bool = True) -> dict:
    """Console status; probe_decider=False skips the TCP check so nothing touches the Decider port."""
    path = replay_path()
    cache = load_replay(path)
    reachable = decider_reachable(CONFIG["decider_url"]) if probe_decider else False
    return {
        "mode": CONFIG["mode"],
        "decider": {"url": CONFIG["decider_url"], "reachable": reachable},
        "pipeline": pipeline_status(),
        "replay": {
            "path": _display(path),
            "frames": len(cache),
            "sources": [str(p.relative_to(ROOT)) for p in replay_sources()],
        },
        "record": CONFIG["record"],
        "pricing_source": _pricing()["source"],
        "reels": REELS,
        "questions": build_questions(),
        "thresholds": asdict(thresholds()),
        "difficulty_note": DIFFICULTY_NOTE if not thresholds().use_difficulty else "used for routing",
    }


@app.get("/api/reel")
def reel(id: str = REELS[0]["id"], mode: str | None = None) -> dict:
    mode = mode or CONFIG["mode"]
    frames, _, source = _resolve_reel(id, mode)
    return {"id": id, "mode": mode, "source": source, "frames": frames}


@app.get("/api/run")
async def run(request: Request, id: str = REELS[0]["id"], mode: str | None = None, pace: float | None = None):
    mode = mode or CONFIG["mode"]
    pace = CONFIG["pace"] if pace is None else max(0.05, min(pace, 5.0))
    frames, cache, source = _resolve_reel(id, mode)
    if mode == "live" and _run_lock.locked():
        raise HTTPException(409, "a live run is already in progress; Decider takes one request at a time")

    async def stream():
        rows: list[dict] = []
        yield _sse("start", {"total": len(frames), "mode": mode, "source": source, "pace": pace})
        if mode == "live":
            await _run_lock.acquire()
        try:
            for meta in frames:
                if await request.is_disconnected():
                    return
                yield _sse("deciding", {"index": meta["index"]})
                if mode == "live":
                    started = time.perf_counter()
                    try:
                        path = IMAGES / meta["file_name"]
                        result = await run_in_threadpool(triage().run, path, meta["label"], meta["lighting"])
                    except Exception as exc:
                        log.exception("live triage failed for %s", meta["file_name"])
                        yield _sse(
                            "failed",
                            {
                                "index": meta["index"],
                                "error": f"{type(exc).__name__}: {exc}"[:240],
                                "elapsed_ms": (time.perf_counter() - started) * 1000,
                            },
                        )
                        continue
                    _record(result)
                else:
                    result = replay_result(meta["file_name"], cache)
                    await asyncio.sleep(_replay_delay(result, pace))
                row = _enrich(result, meta, source)
                rows.append(row)
                yield _sse("frame", row)
            yield _sse("done", _digest(rows))
        finally:
            if mode == "live" and _run_lock.locked():
                _run_lock.release()

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/api/action")
async def action(request: Request) -> JSONResponse:
    body = await request.json()
    act = str(body.get("action", ""))
    if act not in ("dispatch_patrol", "known_staff"):
        raise HTTPException(400, "action must be dispatch_patrol or known_staff")
    frame = _frame_key(body.get("frame"))
    meta = manifest().get(frame or "", {})
    row = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action": act,
        "frame": frame,
        "camera": meta.get("location"),
        "captured": meta.get("captured"),
        "alert": body.get("alert"),
        "escalate": body.get("escalate"),
        "mode": body.get("mode"),
    }
    OUTBOX.mkdir(exist_ok=True)
    with _write_lock, (OUTBOX / "actions.jsonl").open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    return JSONResponse({"ok": True, "logged": row})


@app.get("/thumb/{name}")
def thumb(name: str) -> Response:
    if name not in manifest() or not (IMAGES / name).exists():
        raise HTTPException(404)
    return Response(_thumb_bytes(name), media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@lru_cache(maxsize=1024)
def _thumb_bytes(name: str) -> bytes:
    from PIL import Image

    with Image.open(IMAGES / name) as img:
        img = img.convert("RGB")
        img.thumbnail((360, 360))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=78)
        return buf.getvalue()


app.mount("/frames", StaticFiles(directory=IMAGES), name="frames")
app.mount("/static", StaticFiles(directory=WEB), name="static")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trailgate demo UI")
    parser.add_argument("--mode", choices=["live", "replay"], default=os.environ.get("TRAILGATE_MODE", "replay"))
    parser.add_argument(
        "--replay", help="jsonl of FrameResult rows or Decider answers (default: results/decider_answers.jsonl)"
    )
    parser.add_argument("--record", help="append every live FrameResult to this jsonl, replayable later")
    parser.add_argument("--pace", type=float, default=1.0, help="replay delay multiplier (lower is faster)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    CONFIG.update(mode=args.mode, replay=args.replay, record=args.record, pace=args.pace)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
