"""Deterministic triage: code check, one Decider call per frame, the burst rule, then only the Bedrock calls each tier needs."""

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
    types = {f.name: f.type for f in fields(Thresholds)}
    return Thresholds(**{k: (bool(v) if types[k] in (bool, "bool") else float(v)) for k, v in data.items() if k in types})


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
        return self.run_burst([path], [label], [lighting])[0]

    def run_burst(
        self,
        paths: list[str | Path],
        labels: list[str | None] | None = None,
        lightings: list[str | None] | None = None,
    ) -> list[FrameResult]:
        """Route every frame of one camera burst, apply the burst rule, then call Bedrock per frame by tier.

        A frame is skipped only if no frame in the burst reaches `skip_below`; if any frame
        escalates, the whole burst escalates and the ranger gets one alert for it.
        """
        labels = labels or [None] * len(paths)
        lightings = lightings or [None] * len(paths)
        checks = [frames.check(p) for p in paths]
        data = [frames.load(p) if c.ok else b"" for p, c in zip(paths, checks, strict=True)]
        routed = self.router.decide_burst([self.route(d) for d, c in zip(data, checks, strict=True) if c.ok])
        routes: list[Route | None] = [routed.pop(0) if c.ok else None for c in checks]
        results = [
            self._result(Path(p).name, c, r, label, lighting)
            for p, c, r, label, lighting in zip(paths, checks, routes, labels, lightings, strict=True)
        ]
        escalated = [i for i, r in enumerate(routes) if r is not None and r.tier is Tier.ESCALATE]
        if escalated:
            top = max(escalated, key=lambda i: routes[i].escalate or 0.0)
            self._burst_alert(results[top], str(paths[top]), data[top], routes[top], len(paths))
        if self.bedrock:
            for result, image, route in zip(results, data, routes, strict=True):
                if route is None or route.tier is Tier.SKIP:
                    continue
                if route.tier is Tier.ESCALATE and not self.label_escalations:
                    continue
                species, confidence, call = nova.label_species(image, "small" if route.tier is Tier.SMALL else "large")
                result.species, result.species_confidence = species, confidence
                result.calls.append(call)
        return results

    @staticmethod
    def _result(name: str, check: frames.FrameCheck, route: Route | None, label: str | None, lighting: str | None) -> FrameResult:
        if route is None:
            reason = f"code check: {check.reason}; Decider not asked"
            return FrameResult(name, Tier.SKIP.value, 0.0, None, "n/a", reason, 0.0, label=label, lighting=lighting)
        lighting = lighting or ("night_ir" if check.infrared else "day")
        return FrameResult(
            name, route.tier.value, route.present, route.escalate, route.difficulty, route.reason, route.latency_ms,
            label=label, lighting=lighting,
        )

    def _burst_alert(self, result: FrameResult, path: str, image: bytes, route: Route, size: int) -> None:
        text = ""
        if self.bedrock:
            call = nova.write_alert(image, route)
            result.calls.append(call)
            text = "" if call.error else call.text
        text = text or nova.fallback_alert(result.frame, route)
        if size > 1:
            text = f"{text} (burst of {size} frames)"
        self._alert(result, path, text)

    def run_many(self, paths: list[str | Path]) -> list[FrameResult]:
        return [self.run(p) for p in find_frames(paths)]

    def _alert(self, result: FrameResult, path: str, text: str) -> None:
        result.alert = text
        if self.notify:
            send_alert(path, text, result.present, result.escalate)
