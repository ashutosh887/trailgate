"""The conversational layer: a Strands agent that works through a memory card with PresenceRouter in front of its tools.

"Process tonight's card and tell me what needs me." The agent lists the frames, asks for
a species label on each, and PresenceRouter decides per call: empty frames are denied
before any model runs, the rest get the right-sized Nova model, people and vehicles are
flagged for alert_ranger.

The agent routes frame by frame. The burst rule (skip a frame only if no frame in its
burst reaches skip_below, one alert per escalated burst) lives in `Triage.run_burst`,
which `python -m trailgate run` uses with frames grouped by `trailgate.bursts`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from strands import Agent, tool
from strands.models import BedrockModel

from patterns.presence_router import PresenceRouterCore
from patterns.presence_router.handler import PresenceRouter

from . import bedrock as nova
from . import frames
from .alerts import OUTBOX, send_alert
from .pipeline import find_frames, load_thresholds
from .results import FrameResult

DEFAULT_AGENT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SYSTEM_PROMPT = """You triage camera-trap memory cards for a wildlife ranger.
For a card or folder: call list_frames, then call label_species once for every frame.
If label_species is denied as an empty frame, call archive_empty for that frame.
If a label_species result says escalate, call alert_ranger with a one-line note.
When every frame is handled, call daily_digest and reply with the digest and only the frames that need the ranger.
Never guess a species yourself; use the tool result."""


@dataclass
class CardSession:
    router: PresenceRouter
    labels: dict[str, dict] = field(default_factory=dict)
    alerts: dict[str, str] = field(default_factory=dict)

    def results(self) -> list[FrameResult]:
        rows = []
        for path, route in self.router.routes.items():
            label = self.labels.get(path, {})
            rows.append(FrameResult(
                Path(path).name, route.tier.value, route.present, route.escalate, route.difficulty, route.reason,
                route.latency_ms, species=label.get("species"), species_confidence=label.get("confidence"),
                alert=self.alerts.get(path), calls=label.get("calls", []),
            ))
        return rows


def _precheck(path: Path) -> str | None:
    return frames.check(path).reason


def build_router(core: PresenceRouterCore | None = None) -> PresenceRouter:
    return PresenceRouter(core or PresenceRouterCore(thresholds=load_thresholds()), precheck=_precheck)


def build_tools(session: CardSession) -> list:
    @tool
    def list_frames(folder: str) -> str:
        """List the image files on a memory card or folder.

        Args:
            folder: Path to a folder of camera-trap images.
        """
        return "\n".join(str(p) for p in find_frames([folder])) or "no images found"

    @tool
    def archive_empty(image_path: str) -> str:
        """Archive a frame with nothing in it. No model is called.

        Args:
            image_path: Path to the frame.
        """
        OUTBOX.mkdir(parents=True, exist_ok=True)
        record = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "frame": image_path}
        with open(OUTBOX / "archived.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")
        return f"archived {Path(image_path).name}"

    @tool
    def label_species(image_path: str, model_tier: str = "large", escalate: bool = False) -> str:
        """Label the species in a camera-trap frame with an Amazon Nova model.

        Args:
            image_path: Path to the frame.
            model_tier: "small" (Nova Lite) or "large" (Nova Pro). Set by the router.
            escalate: True when a person or vehicle is likely present. Set by the router.
        """
        key = str(Path(image_path).resolve())
        species, confidence, call = nova.label_species(frames.load(image_path), model_tier)
        session.labels[key] = {"species": species, "confidence": confidence, "calls": [call]}
        outcome = {
            "frame": Path(image_path).name,
            "model": call.model_id,
            "species": species,
            "confidence": confidence,
            "escalate": escalate,
            "error": call.error,
        }
        if escalate:
            outcome["next"] = "escalate: call alert_ranger for this frame"
        return json.dumps(outcome)

    @tool
    def alert_ranger(image_path: str, note: str) -> str:
        """Send a ranger alert for a frame that likely shows a person or a vehicle.

        Args:
            image_path: Path to the frame.
            note: One-line note for the ranger.
        """
        key = str(Path(image_path).resolve())
        route = session.router.routes.get(key)
        record = send_alert(
            image_path, note, route.present if route else None, route.escalate if route else None, source="agent"
        )
        session.alerts[key] = note
        return f"alert sent for {Path(image_path).name}" + (f" (SNS: {record['sns_error']})" if record["sns_error"] else "")

    @tool
    def daily_digest() -> str:
        """Write the morning digest for every frame handled so far."""
        return nova.write_digest(session.results())

    return [list_frames, archive_empty, label_species, alert_ranger, daily_digest]


def build_agent(core: PresenceRouterCore | None = None) -> tuple[Agent, CardSession]:
    session = CardSession(build_router(core))
    model = BedrockModel(
        model_id=os.environ.get("TRAILGATE_AGENT_MODEL", DEFAULT_AGENT_MODEL),
        region_name=nova.region(),
    )
    agent = Agent(
        model=model,
        tools=build_tools(session),
        interventions=[session.router],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
    )
    return agent, session

