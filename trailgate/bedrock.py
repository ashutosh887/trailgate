"""Amazon Bedrock calls for Trailgate: species labels, ranger alerts, the morning digest.

Every call returns a `ModelCall` with tokens, latency and cost. Bedrock errors never
raise: they come back on `ModelCall.error`, so the pipeline keeps running.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from .frames import image_format
from .results import FrameResult, ModelCall

PRICES_SOURCE = "https://aws.amazon.com/bedrock/pricing/ (AWS Price List API, AmazonBedrock, us-east-1, on-demand)"
PRICES: dict[str, tuple[float, float]] = {
    "amazon.nova-lite-v1:0": (0.00006, 0.00024),
    "amazon.nova-pro-v1:0": (0.0008, 0.0032),
}

IMAGE_TOKENS_SOURCE = "https://docs.aws.amazon.com/nova/latest/userguide/modalities-image.html"
NOVA_IMAGE_TOKENS: tuple[tuple[int, int, int], ...] = (
    (900, 450, 800),
    (900, 900, 1300),
    (1400, 900, 1800),
    (1800, 900, 2400),
    (1300, 1300, 2600),
)

SPECIES = (
    "opossum",
    "coyote",
    "rabbit",
    "deer",
    "raccoon",
    "bird",
    "bobcat",
    "cat",
    "car",
    "dog",
    "squirrel",
    "rodent",
    "cow",
    "fox",
    "skunk",
    "lizard",
    "mountain_lion",
    "badger",
    "insect",
    "bat",
    "pig",
)
ANSWERS = (*SPECIES, "empty", "unknown")
SYNONYMS = {
    "mountain lion": "mountain_lion",
    "cougar": "mountain_lion",
    "puma": "mountain_lion",
    "rat": "rodent",
    "mouse": "rodent",
    "mice": "rodent",
    "woodrat": "rodent",
    "gopher": "rodent",
    "vehicle": "car",
    "truck": "car",
    "suv": "car",
    "jeep": "car",
    "hare": "rabbit",
    "jackrabbit": "rabbit",
    "cottontail": "rabbit",
    "none": "empty",
    "nothing": "empty",
    "no animal": "empty",
}
CONFIDENCES = ("high", "medium", "low")

LABEL_PROMPT = (
    "This is one frame from a wildlife camera trap in the American Southwest. "
    "Name what is in it using exactly one class from this list: {classes}. "
    'Use "empty" if no animal or vehicle is visible. '
    'Use "unknown" if an animal is visible but you cannot tell which class it is. '
    "Night frames are infrared and grayscale; look for eye shine and body shape. "
    'Reply with JSON only: {{"species": "<class>", "confidence": "high|medium|low"}}'
)
ALERT_PROMPT = (
    "A camera-trap triage model flagged this frame as likely showing a person or a vehicle "
    "(p={escalate:.2f}). Write one line, at most 25 words, for the ranger on duty: what is in the frame "
    "and where. If you see no person or vehicle, say so plainly. No preamble."
)
DIGEST_PROMPT = (
    "Write the morning digest for a ranger from these camera-trap triage facts. "
    'Start with exactly this sentence: "{headline}" Then at most three short lines: the species seen, '
    "and each frame that needs the ranger with its note. Plain text, no markdown, no invented facts.\n\nFacts:\n{facts}"
)

UNAVAILABLE_RETRY_S = 600.0
_unavailable: dict[str, tuple[str, float]] = {}
_lock = threading.Lock()


def region() -> str:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"


def model_for(tier: str) -> str:
    tier = getattr(tier, "value", tier)
    if tier == "large":
        return os.environ.get("TRAILGATE_LARGE_MODEL", "us.amazon.nova-pro-v1:0")
    return os.environ.get("TRAILGATE_SMALL_MODEL", "us.amazon.nova-lite-v1:0")


def base_model(model_id: str) -> str:
    return re.sub(r"^(us|eu|apac|global|jp|au|ca|us-gov)\.", "", model_id)


def price(model_id: str) -> tuple[float, float]:
    return PRICES.get(base_model(model_id), (0.0, 0.0))


def cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    per_in, per_out = price(model_id)
    return input_tokens / 1000 * per_in + output_tokens / 1000 * per_out


def estimate_image_tokens(width: int, height: int) -> int:
    """Nova v1 image tokens from the closest documented resolution (aspect ratio first, then area)."""
    ratio = max(width, height) / max(1, min(width, height))
    area = width * height

    def distance(row: tuple[int, int, int]) -> tuple[float, float]:
        w, h, _ = row
        return round(abs(ratio - max(w, h) / min(w, h)), 2), abs(area - w * h)

    return min(NOVA_IMAGE_TOKENS, key=distance)[2]


def estimate_text_tokens(text: str) -> int:
    return max(1, len(text) // 4)


LABEL_TEXT = LABEL_PROMPT.format(classes=", ".join(ANSWERS))
LABEL_OUTPUT_TOKENS = 15


def baseline_cost(tier: str, width: int = 1024, height: int = 747) -> float:
    """Estimated $ for one species-label call on one frame: the "Nova on every frame" baseline."""
    tokens_in = estimate_image_tokens(width, height) + estimate_text_tokens(LABEL_TEXT)
    return cost(model_for(tier), tokens_in, LABEL_OUTPUT_TOKENS)


@lru_cache(maxsize=4)
def _client(region_name: str) -> Any:
    import boto3
    from botocore.config import Config

    config = Config(retries={"max_attempts": 2, "mode": "standard"}, connect_timeout=5, read_timeout=60)
    return boto3.client("bedrock-runtime", region_name=region_name, config=config)


def _error_text(exc: Exception) -> str:
    response = getattr(exc, "response", None) or {}
    code = response.get("Error", {}).get("Code") or type(exc).__name__
    message = response.get("Error", {}).get("Message") or str(exc)
    return f"{code}: {message}"


UNRECOVERABLE = ("per day", "accessdenied", "nocredentials", "unable to locate credentials", "expiredtoken")


def _is_quota_exhausted(error: str) -> bool:
    lowered = error.lower()
    return any(marker in lowered for marker in UNRECOVERABLE)


def converse(
    model_id: str,
    tier: str,
    prompt: str,
    image: bytes | None = None,
    max_tokens: int = 64,
    system: str | None = None,
) -> ModelCall:
    with _lock:
        blocked = _unavailable.get(model_id)
    if blocked and time.monotonic() - blocked[1] < UNAVAILABLE_RETRY_S:
        return ModelCall(model_id, tier, 0, 0, 0.0, 0.0, error=f"skipped, Bedrock unavailable: {blocked[0]}")
    content: list[dict] = []
    if image is not None:
        content.append({"image": {"format": image_format(image), "source": {"bytes": image}}})
    content.append({"text": prompt})
    request: dict[str, Any] = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": content}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.0},
    }
    if system:
        request["system"] = [{"text": system}]
    started = time.perf_counter()
    try:
        response = _client(region()).converse(**request)
    except Exception as exc:
        error = _error_text(exc)
        if _is_quota_exhausted(error):
            with _lock:
                _unavailable[model_id] = (error, time.monotonic())
        return ModelCall(model_id, tier, 0, 0, (time.perf_counter() - started) * 1000, 0.0, error=error)
    latency = response.get("metrics", {}).get("latencyMs") or (time.perf_counter() - started) * 1000
    usage = response.get("usage", {})
    tokens_in, tokens_out = int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))
    text = "".join(block.get("text", "") for block in response["output"]["message"]["content"]).strip()
    return ModelCall(model_id, tier, tokens_in, tokens_out, float(latency), cost(model_id, tokens_in, tokens_out), text)


def normalise_species(raw: str) -> str | None:
    value = raw.strip().lower().replace("-", " ")
    if value in SYNONYMS:
        return SYNONYMS[value]
    value = value.replace(" ", "_")
    if value in ANSWERS:
        return value
    if value.endswith("s") and value[:-1] in ANSWERS:
        return value[:-1]
    return None


def parse_label(text: str) -> tuple[str, str]:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            species = normalise_species(str(data.get("species", "")))
            confidence = str(data.get("confidence", "low")).strip().lower()
            if species:
                return species, confidence if confidence in CONFIDENCES else "low"
        except (json.JSONDecodeError, AttributeError):
            pass
    lowered = text.lower()
    for phrase, species in (*SYNONYMS.items(), *((s.replace("_", " "), s) for s in ANSWERS)):
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            return species, "low"
    return "unknown", "low"


def label_species(image: bytes, tier: str) -> tuple[str | None, str | None, ModelCall]:
    tier = getattr(tier, "value", tier)
    call = converse(model_for(tier), tier, LABEL_TEXT, image, max_tokens=40)
    if call.error:
        return None, None, call
    species, confidence = parse_label(call.text)
    return species, confidence, call


def write_alert(image: bytes, route: Any) -> ModelCall:
    escalate = getattr(route, "escalate", None) or 0.0
    return converse(model_for("small"), "alert", ALERT_PROMPT.format(escalate=escalate), image, max_tokens=60)


def fallback_alert(frame: str, route: Any) -> str:
    escalate = getattr(route, "escalate", None) or 0.0
    return f"Possible person or vehicle in {frame} (Decider p={escalate:.2f}); check the frame."


def digest_facts(results: Iterable[FrameResult]) -> dict:
    rows = list(results)
    tiers = Counter(r.tier for r in rows)
    species = Counter(r.species for r in rows if r.species and r.species not in ("empty", "unknown"))
    escalations = [{"frame": r.frame, "note": r.alert or r.reason} for r in rows if r.tier == "escalate"]
    labelled = sum(1 for r in rows if r.tier in ("small", "large", "escalate") and r.species)
    return {
        "frames": len(rows),
        "skipped": tiers.get("skip", 0),
        "labelled": labelled,
        "for_you": len(escalations),
        "small": tiers.get("small", 0),
        "large": tiers.get("large", 0),
        "species": dict(species.most_common()),
        "escalations": escalations,
        "cost_usd": round(sum(r.cost_usd for r in rows), 6),
    }


def digest_headline(facts: dict) -> str:
    return (
        f"{facts['frames']} frames: {facts['skipped']} empty skipped, "
        f"{facts['labelled']} labelled, {facts['for_you']} for you."
    )


def plain_digest(facts: dict) -> str:
    lines = [digest_headline(facts)]
    if facts["species"]:
        lines.append(
            "Seen: " + ", ".join(f"{name.replace('_', ' ')} x{n}" for name, n in facts["species"].items()) + "."
        )
    lines.extend(f"Check {e['frame']}: {e['note']}" for e in facts["escalations"])
    lines.append(f"Bedrock spend: ${facts['cost_usd']:.4f}.")
    return "\n".join(lines)


def write_digest(results: Iterable[FrameResult]) -> str:
    facts = digest_facts(results)
    prompt = DIGEST_PROMPT.format(headline=digest_headline(facts), facts=json.dumps(facts, indent=1))
    call = converse(model_for("small"), "digest", prompt, max_tokens=200)
    return plain_digest(facts) if call.error or not call.text else call.text
