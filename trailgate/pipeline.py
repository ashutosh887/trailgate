"""Deterministic triage for one frame: code check, one Decider call, then only the Bedrock call the tier needs."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from patterns.presence_router import DeciderClient, PresenceRouterCore, Route, Thresholds, Tier

from . import bedrock as nova
from . import frames
from .alerts import send_alert
from .results import FrameResult

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS_FILE = ROOT / "results" / "thresholds.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_thresholds(path: Path = THRESHOLDS_FILE) -> Thresholds:
    if not path.exists():
        return Thresholds()
    data = json.loads(path.read_text())
    known = {f.name for f in fields(Thresholds)}
    return Thresholds(**{k: float(v) for k, v in data.items() if k in known})


def find_frames(paths: list[str | Path]) -> list[Path]:
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES))
        else:
            found.append(p)
    return found


class Triage:
    """Route each frame to skip, small, large or escalate, and call Bedrock only where the tier needs it."""

    def __init__(
        self,
        bedrock: bool = True,
        thresholds: Thresholds | None = None,
        router: PresenceRouterCore | None = None,
        notify: bool = True,
        label_escalations: bool = True,
    ):
        self.bedrock = bedrock
        self.router = router or PresenceRouterCore(thresholds=thresholds or load_thresholds(), client=DeciderClient())
        self.notify = notify
        self.label_escalations = label_escalations

    @property
    def thresholds(self) -> Thresholds:
        return self.router.thresholds

    def route(self, data: bytes) -> Route:
        try:
            return self.router.route(data)
        except Exception as exc:
            reason = f"Decider unavailable ({type(exc).__name__}); failing open to the large model"
            return Route(Tier.LARGE, 0.5, None, "unknown", 0.0, reason, 0.0)

    def run(self, path: str | Path, label: str | None = None, lighting: str | None = None) -> FrameResult:
        name = Path(path).name
        check = frames.check(path)
        if not check.ok:
            reason = f"code check: {check.reason}; Decider not asked"
            return FrameResult(name, Tier.SKIP.value, 0.0, None, "n/a", reason, 0.0, label=label, lighting=lighting)
        data = frames.load(path)
        lighting = lighting or ("night_ir" if check.infrared else "day")
        route = self.route(data)
        result = FrameResult(
            name, route.tier.value, route.present, route.escalate, route.difficulty, route.reason, route.latency_ms,
            label=label, lighting=lighting,
        )
        if not self.bedrock or route.tier is Tier.SKIP:
            if route.tier is Tier.ESCALATE:
                self._alert(result, str(path), nova.fallback_alert(name, route))
            return result
        if route.tier is Tier.ESCALATE:
            call = nova.write_alert(data, route)
            result.calls.append(call)
            self._alert(result, str(path), call.text if not call.error and call.text else nova.fallback_alert(name, route))
            if not self.label_escalations:
                return result
        tier = "small" if route.tier is Tier.SMALL else "large"
        species, confidence, call = nova.label_species(data, tier)
        result.species, result.species_confidence = species, confidence
        result.calls.append(call)
        return result

    def run_many(self, paths: list[str | Path]) -> list[FrameResult]:
        return [self.run(p) for p in find_frames(paths)]

    def _alert(self, result: FrameResult, path: str, text: str) -> None:
        result.alert = text
        if self.notify:
            send_alert(path, text, result.present, result.escalate)
