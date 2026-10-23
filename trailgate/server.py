"""Trailgate demo server: a night's camera-trap card, triaged frame by frame over SSE.

Modes:
  live    each frame goes through trailgate.pipeline.Triage (Decider at 127.0.0.1:8099, then Bedrock)
  replay  results are read from a cached jsonl in results/ and paced like a live run

Run:
  python -m trailgate.server   (replays results/decider_answers.jsonl with the v2 burst policy)
  python -m trailgate.server --replay results/decider_answers.jsonl
  python -m trailgate.server --mode live --record results/run-live.jsonl

Bedrock spend is the real `ModelCall.cost_usd` when calls were made, otherwise the estimate from
`bedrock.tier_cost` per frame plus one `bedrock.alert_cost` per escalated burst, labelled "est." in the UI.
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
from starlette.background import BackgroundTask

from patterns.presence_router import PresenceRouterCore, Route, Thresholds, build_questions
from patterns.presence_router.router import DEFAULT_URL
from trailgate import bedrock
from trailgate.pipeline import Triage, load_thresholds
from trailgate.results import FrameResult

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent / "web"
IMAGES = ROOT / "data" / "images"
MANIFESTS = ROOT / "data"
RESULTS = ROOT / "results"
OUTBOX = Path(os.environ.get("TRAILGATE_OUTBOX", ROOT / "outbox"))

TIER_MODELS = {"small": "Nova Lite", "large": "Nova Pro", "escalate": "Nova Pro label + Nova Lite alert"}
SETUP_HINT = "No frames in data/images: run scripts/fetch_cct.py and scripts/fetch_bursts.py first"

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


def _model_name(model_id: str) -> str:
    return "Nova Pro" if "pro" in model_id.lower() else "Nova Lite"


def _line(model: str, kind: str, usd: float, frame: str, estimated: bool, call: dict | None = None) -> dict:
    call = call or {}
    return {
        "model": model,
        "kind": kind,
        "usd": usd,
        "estimated": estimated,
        "frame": frame,
        "latency_ms": call.get("latency_ms"),
        "error": call.get("error"),
    }


def costing(result: dict, charge_alert: bool) -> dict:
    """Bedrock lines for a row: real calls when made (errored ones estimated), else estimates by tier.

    `charge_alert` adds the one Nova Lite alert of an escalated burst when no call was recorded for it.
    Baselines cover the same frames as the lines: one frame in replay, the whole burst in live mode.
    """
    frame = result["frame"]
    lines = []
    for call in result.get("calls") or []:
        kind = "alert" if call["tier"] == "alert" else "label"
        estimated = bool(call.get("error"))
        usd = (
            (bedrock.alert_cost() if kind == "alert" else bedrock.tier_cost(call["tier"]))
            if estimated
            else call["cost_usd"]
        )
        lines.append(_line(_model_name(call["model_id"]), kind, usd, call.get("frame", frame), estimated, call))
    tier = result["tier"]
    if not result.get("calls") and tier != "skip":
        label_model = TIER_MODELS["large" if tier == "escalate" else tier]
        lines.append(_line(label_model, "label", bedrock.tier_cost(tier), frame, True))
        if tier == "escalate" and charge_alert:
            lines.append(_line("Nova Lite", "alert", bedrock.alert_cost(), frame, True))
    frames = int(result.get("burst_frames") or 1)
    return {
        "actual_usd": sum(x["usd"] for x in lines if not x["estimated"]),
        "estimated_usd": sum(x["usd"] for x in lines if x["estimated"]),
        "frames": frames,
        "baseline_usd": bedrock.tier_cost("large") * frames,
        "baseline_lite_usd": bedrock.tier_cost("small") * frames,
        "lines": lines,
    }


def _frame_key(value: str | None) -> str | None:
    return Path(value).name if value else None


def _burst_key(file_name: str) -> str:
    return manifest().get(file_name, {}).get("seq_id") or file_name


def _normalise(row: dict) -> tuple[str, dict]:
    """A recorded live row (from --record) as is, or a results/decider_answers.jsonl row as a FrameResult dict."""
    if "tier" in row:
        return row["frame"], {**row, "kind": "recorded"}
    key = row["file_name"]
    route = _core().decide(row["answers"], float(row.get("server_latency_ms") or 0.0))
    meta = manifest().get(key, {})
    result = FrameResult(
        frame=key,
        tier=route.tier.value,
        present=route.present,
        escalate=route.escalate,
        difficulty=route.difficulty,
        reason=route.reason,
        decider_latency_ms=route.latency_ms,
        label=meta.get("label"),
        lighting=meta.get("lighting"),
    ).to_dict()
    result["answers"] = route.answers
    result["difficulty_confidence"] = route.difficulty_confidence
    result["kind"] = "answers"
    return key, result


@lru_cache(maxsize=1)
def thresholds() -> Thresholds:
    return load_thresholds()


@lru_cache(maxsize=1)
def _core() -> PresenceRouterCore:
    return PresenceRouterCore(thresholds=thresholds())


def replay_path() -> Path:
    return Path(CONFIG["replay"]) if CONFIG["replay"] else DEFAULT_REPLAY


def load_replay(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    stat = path.stat()
    return _load_replay_cached(str(path), stat.st_mtime, stat.st_size)


@lru_cache(maxsize=8)
def _load_replay_cached(path: str, mtime: float, size: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                key, result = _normalise(json.loads(line))
                if (IMAGES / key).exists():
                    out[key] = result
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


class ConsoleTriage(Triage):
    """The pipeline without its own notifier (the console's phone card is the alert surface), keeping each Route."""

    def __init__(self):
        super().__init__(bedrock=True, notify=False)
        self.routes: list[Route] = []

    def route(self, data: bytes) -> Route:
        route = super().route(data)
        self.routes.append(route)
        return route


_triage: ConsoleTriage | None = None


def triage() -> ConsoleTriage:
    global _triage
    if _triage is None:
        _triage = ConsoleTriage()
    return _triage


def live_result(path: Path) -> dict:
    """Run the frame's whole burst; the frame carries the burst's Bedrock calls, cost and alert."""
    names = [f["file_name"] for f in burst_of(path.name) if (IMAGES / f.get("file_name", "")).is_file()]
    if path.name not in names:
        names = [path.name]
    metas = [manifest().get(n, {}) for n in names]
    t = triage()
    t.routes = []
    results = t.run_burst(
        [IMAGES / n for n in names], [m.get("label") for m in metas], [m.get("lighting") for m in metas]
    )
    at = names.index(path.name)
    data = results[at].to_dict()
    data["calls"] = [{**c, "frame": r["frame"]} for r in (x.to_dict() for x in results) for c in r["calls"]]
    data["cost_usd"] = sum(r.cost_usd for r in results)
    data["alert"] = next((r.alert for r in results if r.alert), None)
    data["burst_frames"] = len(names)
    if len(t.routes) == len(names):
        data["answers"] = t.routes[at].answers
        data["difficulty_confidence"] = t.routes[at].difficulty_confidence
    if len(names) > 1:
        data["burst"] = [
            {
                "file_name": r.frame,
                "present": r.present,
                "escalate": r.escalate,
                "self": i == at,
                "thumb": f"/thumb/{r.frame}",
            }
            for i, r in enumerate(results)
        ]
    return data


def _record(row: dict) -> None:
    if not CONFIG["record"]:
        return
    path = Path(CONFIG["record"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock, path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def _enrich(result: dict, meta: dict, source: str, alerted: set[str]) -> dict:
    """A streamed row; `alerted` holds the bursts whose alert is already charged in this run."""
    result.setdefault("difficulty_confidence", (result.get("answers") or {}).get("difficulty", {}).get("confidence"))
    burst = _burst_key(meta["file_name"])
    charge_alert = result["tier"] == "escalate" and burst not in alerted
    if charge_alert:
        alerted.add(burst)
    return {
        "index": meta["index"],
        "meta": meta,
        "result": result,
        "model": TIER_MODELS.get(result["tier"]),
        "costing": costing(result, charge_alert),
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
        "bedrock_calls": sum(len(r["costing"]["lines"]) for r in rows),
        "alert_frames": [r["meta"] for r in rows if r["result"]["tier"] == "escalate"],
    }


def _resolve_reel(reel_id: str, mode: str) -> tuple[list[dict], dict[str, dict], str]:
    reel = next((r for r in REELS if r["id"] == reel_id), REELS[0])
    if mode == "replay":
        path = replay_path()
        cache = load_replay(path)
        frames = build_reel(reel["seed"], reel["size"], cache) if cache else []
        return frames, cache, _display(path)
    return build_reel(reel["seed"], reel["size"]), {}, "live"


def _display(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/status")
def status() -> dict:
    return status_info()


def setup_message(replay_frames: int) -> str | None:
    if not any(IMAGES.glob("*.jpg")):
        return SETUP_HINT
    if not replay_frames:
        return f"No replayable frames in {_display(replay_path())}"
    return None


def status_info(probe_decider: bool = True) -> dict:
    """Console status; probe_decider=False skips the TCP check so nothing touches the Decider port."""
    path = replay_path()
    frames = len(load_replay(path))
    reachable = decider_reachable(CONFIG["decider_url"]) if probe_decider else False
    return {
        "mode": CONFIG["mode"],
        "decider": {"url": CONFIG["decider_url"], "reachable": reachable},
        "replay": {"path": _display(path), "frames": frames},
        "setup": setup_message(frames),
        "record": CONFIG["record"],
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
    held = False
    if mode == "live":
        if _run_lock.locked():
            raise HTTPException(409, "a live run is already in progress; Decider takes one request at a time")
        await _run_lock.acquire()
        held = True

    def release() -> None:
        nonlocal held
        if held:
            held = False
            _run_lock.release()

    async def stream():
        rows: list[dict] = []
        alerted: set[str] = set()
        try:
            yield _sse("start", {"total": len(frames), "mode": mode, "source": source, "pace": pace})
            for meta in frames:
                if await request.is_disconnected():
                    return
                yield _sse("deciding", {"index": meta["index"]})
                if mode == "live":
                    started = time.perf_counter()
                    try:
                        result = await run_in_threadpool(live_result, IMAGES / meta["file_name"])
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
                row = _enrich(result, meta, source, alerted)
                rows.append(row)
                yield _sse("frame", row)
            yield _sse("done", _digest(rows))
        finally:
            release()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        background=BackgroundTask(release),
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


def _image(name: str) -> Path:
    if name not in manifest() or not (IMAGES / name).is_file():
        raise HTTPException(404)
    return IMAGES / name


@app.get("/frames/{name}")
def frame(name: str) -> FileResponse:
    return FileResponse(_image(name), headers={"Cache-Control": "max-age=86400"})


@app.get("/thumb/{name}")
def thumb(name: str) -> Response:
    _image(name)
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


app.mount("/static", StaticFiles(directory=WEB), name="static")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trailgate demo UI")
    parser.add_argument("--mode", choices=["live", "replay"], default=os.environ.get("TRAILGATE_MODE", "replay"))
    parser.add_argument(
        "--replay", help="Decider answers (results/decider_answers.jsonl, the default) or rows recorded with --record"
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
