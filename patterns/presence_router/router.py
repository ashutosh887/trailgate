"""Framework-free routing: one Decider call per image, a tier out.

Tiers, cheapest first:
  skip      nothing worth a model's time; no model is called
  small     the target is clearly present; a small model is enough
  large     Decider is unsure; use the stronger model
  escalate  the escalation subject is present; a person should see it now

The default policy is a confidence band on P(target). `use_difficulty=True` restores the
v1 policy, where the small tier also needed Decider to call the subject "clear".
`route_burst` routes the frames of one camera burst together: a frame is skipped only if
no frame in the burst reaches `skip_below`, and the burst escalates if any frame does.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path

from .questions import build_questions

DEFAULT_URL = os.environ.get("STRANDS_DECIDER_URL", "http://127.0.0.1:8099")


class Tier(str, Enum):
    SKIP = "skip"
    SMALL = "small"
    LARGE = "large"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class Thresholds:
    skip_below: float = 0.35
    small_above: float = 0.90
    escalate_above: float = 0.50
    use_difficulty: bool = False


@dataclass
class Route:
    tier: Tier
    present: float
    escalate: float | None
    difficulty: str
    difficulty_confidence: float
    reason: str
    latency_ms: float
    answers: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tier"] = self.tier.value
        return data


class DeciderClient:
    """Minimal client for `strands-decider serve --vision`. Calls are serialised:
    the reference server is not safe under concurrent requests."""

    _lock = threading.Lock()

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def ask(self, questions: dict, images: list[bytes], state: str = "") -> dict:
        body = {
            "state": state,
            "questions": questions,
            "images": [base64.b64encode(b).decode() for b in images],
        }
        req = urllib.request.Request(
            f"{self.url}/v1/systemone",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with self._lock:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())


class PresenceRouterCore:
    def __init__(
        self,
        target: str = "an animal",
        escalate: str | None = "a person or a vehicle",
        thresholds: Thresholds | None = None,
        client: DeciderClient | None = None,
        state: str = "A single frame from a wildlife camera trap.",
    ):
        self.questions = build_questions(target, escalate)
        self.thresholds = thresholds or Thresholds()
        self.client = client or DeciderClient()
        self.state = state

    def decide(self, answers: dict, latency_ms: float = 0.0) -> Route:
        t = self.thresholds
        present = answers["present"]["noul"]
        escalate = answers["escalate"]["noul"] if "escalate" in answers else None
        difficulty = answers["difficulty"]["choice"]
        difficulty_conf = answers["difficulty"]["confidence"]
        if escalate is not None and escalate >= t.escalate_above:
            tier, reason = Tier.ESCALATE, f"escalation subject at {escalate:.2f} >= {t.escalate_above}"
        elif present < t.skip_below:
            tier, reason = Tier.SKIP, f"target at {present:.2f} < {t.skip_below}; no model called"
        elif present >= t.small_above and (difficulty == "clear" or not t.use_difficulty):
            tier, reason = Tier.SMALL, f"target at {present:.2f} >= {t.small_above}"
        elif t.use_difficulty and present >= t.small_above:
            tier, reason = Tier.LARGE, f"target at {present:.2f}, difficulty {difficulty}"
        else:
            tier, reason = Tier.LARGE, f"target at {present:.2f}: Decider is unsure"
        return Route(tier, present, escalate, difficulty, difficulty_conf, reason, latency_ms, answers)

    def route(self, image: bytes | str | Path) -> Route:
        data = Path(image).read_bytes() if isinstance(image, (str, Path)) else image
        started = time.perf_counter()
        response = self.client.ask(self.questions, [data], self.state)
        return self.decide(response["answers"], (time.perf_counter() - started) * 1000)

    def decide_burst(self, routes: list[Route]) -> list[Route]:
        if not routes:
            return routes
        t = self.thresholds
        top = max(r.present for r in routes)
        escalations = [r.escalate for r in routes if r.escalate is not None]
        top_escalate = max(escalations) if escalations else None
        out = []
        for r in routes:
            if top_escalate is not None and top_escalate >= t.escalate_above and r.tier is not Tier.ESCALATE:
                r = replace(
                    r, tier=Tier.ESCALATE, reason=f"burst: escalation subject at {top_escalate:.2f} in another frame"
                )
            elif r.tier is Tier.SKIP and top >= t.skip_below:
                r = replace(
                    r,
                    tier=Tier.LARGE,
                    reason=f"burst: target at {top:.2f} in another frame; this frame {r.present:.2f}",
                )
            out.append(r)
        return out

    def route_burst(self, images: list[bytes | str | Path]) -> list[Route]:
        return self.decide_burst([self.route(image) for image in images])


def route(image: bytes | str | Path, **kwargs) -> Route:
    return PresenceRouterCore(**kwargs).route(image)
