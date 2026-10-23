"""Typed Decider questions for PresenceRouter.

One call asks every question about the same image, so the image is encoded once.
`target` is the thing an agent cares about: "an animal", "a parcel", "a defect".
"""

from __future__ import annotations

DIFFICULTY_OPTIONS = {
    "clear": "the subject is large, lit and in focus",
    "hard": "the subject is small, dark, blurred, infrared or partly hidden",
}


def presence_question(target: str = "an animal", hint: str = "even partly hidden or at the edge") -> dict:
    return {
        "type": "noul",
        "instructions": f"Is there {target} anywhere in this photo, {hint}?",
        "criteria": {
            "true": f"{target} or part of one is visible",
            "false": "only vegetation, ground, sky, or camera artefacts",
        },
    }


def escalate_question(subject: str = "a person or a vehicle") -> dict:
    return {"type": "noul", "instructions": f"Is there {subject} in this photo?"}


def difficulty_question() -> dict:
    return {"type": "choice", "instructions": "How easy is the subject to see?", "criteria": DIFFICULTY_OPTIONS}


def build_questions(target: str = "an animal", escalate: str | None = "a person or a vehicle") -> dict:
    questions = {"present": presence_question(target), "difficulty": difficulty_question()}
    if escalate:
        questions["escalate"] = escalate_question(escalate)
    return questions
