# PresenceRouter

This is Trailgate's reusable decision pattern, for any Strands agent that sends images to an expensive model. Before the tool runs, Strands Decider 2B answers one question, whether the thing this agent cares about is in the image at all, and the answer picks a tier.

| Tier | When | What happens |
|---|---|---|
| `escalate` | P(escalation subject) at or above `escalate_above` | `Transform`: `model_tier="large"`, `escalate=True` |
| `skip` | P(target) below `skip_below` | `Deny`: the tool never runs and no model is called |
| `small` (Nova Lite · Decider confident in Trailgate) | P(target) at or above `small_above` | `Transform`: `model_tier="small"` |
| `large` (Nova Pro · Decider unsure in Trailgate) | anything in between | `Transform`: `model_tier="large"` |

The tiers are a confidence band on one probability, and the `large` tier is how low confidence is handled. Decider is also asked how hard the subject is to see, but it called 84% of held-out camera-trap frames "hard" and P(hard) separated night from day with AUC 0.67, so the default policy does not route on it. `Thresholds(use_difficulty=True)` restores the earlier v1 policy, where `small` also needed a "clear" answer.

All the questions go to Decider in one call, so the image is encoded once. On Trailgate's tuning set, asking the same wording in a 9-question call and in the 3-question production call changed P(animal) by at most 0.0041.

## The questions

From `questions.py`, with `target="an animal"` and `escalate="a person or a vehicle"`. `noul` is Decider's yes/no type and returns P(true). `choice` picks one option from `criteria`.

```python
{
    "present": {
        "type": "noul",
        "instructions": "Is there an animal anywhere in this photo, even partly hidden or at the edge?",
        "criteria": {
            "true": "an animal or part of one is visible",
            "false": "only vegetation, ground, sky, or camera artefacts",
        },
    },
    "difficulty": {
        "type": "choice",
        "instructions": "How easy is the subject to see?",
        "criteria": {
            "clear": "the subject is large, lit and in focus",
            "hard": "the subject is small, dark, blurred, infrared or partly hidden",
        },
    },
    "escalate": {
        "type": "noul",
        "instructions": "Is there a person or a vehicle in this photo?",
        "criteria": {
            "true": "a person, car, truck, motorbike or other vehicle, or part of one, is visible",
            "false": "only animals, vegetation, ground, sky, or camera artefacts",
        },
    },
}
```

`target` can be anything you can name: "a parcel or package", "a defect", "a document". Set `escalate=None` to drop the third question.

## Drop-in

```python
from strands import Agent, tool
from patterns.presence_router import PresenceRouterCore, Thresholds
from patterns.presence_router.handler import PresenceRouter


@tool
def describe_image(image_path: str, model_tier: str = "large") -> str:
    """Describe the image with the model the router picked."""
    ...


router = PresenceRouter(
    PresenceRouterCore(target="a parcel or package", escalate=None, thresholds=Thresholds(skip_below=0.3)),
    tools={"describe_image"},
)
agent = Agent(tools=[describe_image], interventions=[router])
```

The router reads the image path from `image_path` and writes the tier to `model_tier`; `image_arg` and `tier_arg` rename them. It only writes arguments the tool's schema declares. `router.last_route` holds the latest `Route` (tier, probabilities, difficulty, Decider latency) for display, and `router.routes` holds every routed image by resolved path, including images `precheck` rejected and images Decider failed on. `examples/parcel_photos.py` is a runnable version.

Without Strands, call `PresenceRouterCore(...).route(path)` directly and get the same `Route` back.

## Bursts

Camera traps fire in bursts of 1 to 5 frames, and an animal is often in only one of them. `core.route_burst([frame1, frame2, frame3])` routes each frame with its own Decider call, then applies the burst rule. A frame is skipped only if no frame in the burst reaches `skip_below`, and every frame escalates if any frame does. `core.decide_burst(routes)` does the same on routes you already have. A frame Decider failed on stays on the large tier and is left out of the burst maxima, so it never promotes or escalates its siblings. In Trailgate the burst rule cut missed animal frames from 23 to 9 of 271 at the same thresholds.

One multi-image call per burst was faster (p50 621 ms against 1,473 ms for one call per frame) and agreed with max-over-frames on all 29 multi-frame tuning bursts at the skip threshold. Only one of those bursts was empty, too few to show it separates empty from animal as well, so it is not the default.

## Set thresholds from your own images

The defaults (`skip_below=0.35`, `small_above=0.90`, `escalate_above=0.50`) are a starting point. Trailgate's own frozen values (0.41, 0.72, 0.54) are in `results/thresholds.json`. Label about 50 of your own images, from cameras or sources you will not evaluate on, then run:

```bash
python -m patterns.presence_router.tune_questions --manifest your_labels.csv --negative empty
```

It scores several wordings of each question in one Decider call per image and prints, for each wording, how well it separates positives from negatives and the highest skip threshold at which no positive image is skipped. Pick the wording, take a skip threshold a little below that zero-miss value, and pass it as `Thresholds(...)` (Trailgate reads `results/thresholds.json`). Check the result on a held-out set before you trust it.

## Failure behaviour

If Decider is down, times out or returns something unexpected, `core.try_route` returns a `Route` on the `large` tier with `decider_error` set to the exception, and the handler sends the call to the large tier with a `Transform`, whatever the tool's own default. The expensive model still runs, so a broken router costs money but never loses an image. A failed route's `present` is a placeholder of 0.0; check `decider_error` before reading it. `core.route` still raises, for callers that want to handle the error themselves. The handler retries Decider for an image whose last route failed.

Pass `precheck=` to reject unreadable images in code before Decider sees them. A rejected image is denied and recorded in `router.routes` as a `skip` whose reason starts with `code check:`.

## Limits

- Decider is zero-shot on images. It was not trained on vision tasks, so measure it on your images before relying on the thresholds. Trailgate's presence AUC fell from 0.954 on tuning cameras to 0.900 on unseen ones.
- It stays confident on a blank or black image. Check in code that the image exists, decodes and is not uniform before Decider sees it (`precheck`).
- Small, dark, blurred and night infrared subjects score lower. Report day and night accuracy separately, and keep the skip threshold low enough that hard frames go to the large model rather than to `skip`.
- The probabilities are not a guarantee. Above 0.9 the model card reports about 95% accuracy on its own benchmarks, not on yours.
- The reference server is not safe under concurrent requests, so `DeciderClient` serialises calls within a process.
