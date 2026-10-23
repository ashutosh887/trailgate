"""Ranger alert sink: a local JSONL outbox, plus Amazon SNS when a topic is configured."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

OUTBOX = Path(os.environ.get("TRAILGATE_OUTBOX", Path(__file__).resolve().parent.parent / "outbox"))


def send_alert(
    frame: str,
    text: str,
    present: float | None = None,
    escalate: float | None = None,
    source: str = "pipeline",
) -> dict:
    record = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frame": frame,
        "alert": text,
        "present": present,
        "escalate": escalate,
        "source": source,
        "sns_message_id": None,
        "sns_error": None,
    }
    topic = os.environ.get("TRAILGATE_ALERT_TOPIC_ARN")
    if topic:
        record.update(_publish(topic, record))
    OUTBOX.mkdir(parents=True, exist_ok=True)
    with open(OUTBOX / "alerts.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def _publish(topic: str, record: dict) -> dict:
    import boto3

    scores = ", ".join(
        f"{name} p={value:.2f}" for name, value in (("person/vehicle", record["escalate"]), ("animal", record["present"]))
        if value is not None
    )
    message = f"{record['alert']}\n\nFrame: {record['frame']}\nDecider: {scores}\nAt: {record['at']}"
    subject = f"Trailgate: check {Path(record['frame']).name}"[:100]
    try:
        region = topic.split(":")[3] or None
        response = boto3.client("sns", region_name=region).publish(TopicArn=topic, Subject=subject, Message=message)
        return {"sns_message_id": response.get("MessageId")}
    except Exception as exc:
        return {"sns_error": f"{type(exc).__name__}: {exc}"}
