from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ModelCall:
    model_id: str
    tier: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    text: str = ""
    error: str | None = None


@dataclass
class FrameResult:
    frame: str
    tier: str
    present: float
    escalate: float | None
    difficulty: str
    reason: str
    decider_latency_ms: float
    species: str | None = None
    species_confidence: str | None = None
    alert: str | None = None
    calls: list[ModelCall] = field(default_factory=list)
    label: str | None = None
    lighting: str | None = None

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cost_usd"] = self.cost_usd
        return data
