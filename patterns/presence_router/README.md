# PresenceRouter

A Strands intervention for agents that send images to an expensive model. Before the tool runs, Strands Decider 2B answers one question: is the thing this agent cares about in the image at all? The answer picks a tier:

| Tier | When | What happens |
|---|---|---|
| `escalate` | P(escalation subject) at or above `escalate_above` | `Transform`: `model_tier="large"`, `escalate=True` |
| `skip` | P(target) below `skip_below` | `Deny`: the tool never runs and no model is called |
| `small` | P(target) at or above `small_above` | `Transform`: `model_tier="small"` |
| `large` | anything in between: Decider is unsure | `Transform`: `model_tier="large"` |

The tiers are a confidence band on one probability. Decider is also asked how hard the subject is to see, but on camera-trap frames it called 84% of held-out frames "hard" and P(hard) separated night from day with AUC 0.67, so the default policy does not route on it. `Thresholds(use_difficulty=True)` restores the earlier policy, where `small` also needed a "clear" answer.

All the questions go to Decider in one call, so the image is encoded once. A batch of questions about the same image costs about the same as one question.

## The questions

From `questions.py`, with `target="an animal"` and `escalate="a person or a vehicle"`:

```python
{
  "present": {"type": "noul",
              "instructions": "Is there an animal anywhere in this photo, even partly hidden or at the edge?",
              "criteria": {"true": "an animal or part of one is visible",
                           "false": "only vegetation, ground, sky, or camera artefacts"}},
  "difficulty": {"type": "choice", "instructions": "How easy is the subject to see?",
                 "criteria": {"clear": "the subject is large, lit and in focus",
                              "hard": "the subject is small, dark, blurred, infrared or partly hidden"}},
  "escalate": {"type": "noul", "instructions": "Is there a person or a vehicle in this photo?",
               "criteria": {"true": "a person, car, truck, motorbike or other vehicle, or part of one, is visible",
                            "false": "only animals, vegetation, ground, sky, or camera artefacts"}},
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

router = PresenceRouter(PresenceRouterCore(target="a parcel or package", escalate=None,
                                           thresholds=Thresholds(skip_below=0.3)),
                        tools={"describe_image"})
agent = Agent(tools=[describe_image], interventions=[router])
```

The router reads the image path from `image_path` and writes the tier to `model_tier`; `image_arg` and `tier_arg` rename them. It only writes arguments the tool's schema declares. `router.last_route` holds the latest `Route` (tier, probabilities, difficulty, Decider latency) for display. `examples/parcel_photos.py` is a runnable version.

Without Strands, call `PresenceRouterCore(...).route(path)` directly and get the same `Route` back.

## Bursts

Camera traps fire in bursts of 1 to 5 frames, and an animal is often in only one of them. `core.route_burst([frame1, frame2, frame3])` routes each frame with its own Decider call, then applies burst context: a frame is skipped only if no frame in the burst reaches `skip_below`, and every frame escalates if any frame does. `core.decide_burst(routes)` does the same on routes you already have. Asking one multi-image call per burst was faster (p50 621 ms against 1,473 ms for one call per frame) and agreed with max-over-frames on all 29 multi-frame tuning bursts at the skip threshold, but those bursts held only one empty burst, too few to show it separates empty from animal as well, so it is not the default.

## Set thresholds from your own images

The defaults (`skip_below=0.35`, `small_above=0.90`, `escalate_above=0.50`) are a starting point; Trailgate's own frozen values are in `results/thresholds.json`. Label about 50 of your own images, from cameras or sources you will not evaluate on, then run:

```bash
python -m patterns.presence_router.tune_questions --manifest your_labels.csv --negative empty
```

It scores several wordings of each question in one Decider call per image and prints, for each wording, how well it separates positives from negatives and the highest skip threshold at which no positive image is skipped. Pick the wording, take a skip threshold a little below that zero-miss value, and pass it as `Thresholds(...)` (Trailgate reads `results/thresholds.json`). Check the result on a held-out set before you trust it.

## Failure behaviour

`on_error = "proceed"`: if Decider is down, times out or returns something unexpected, the handler is skipped and the tool runs with its own default tier. The expensive model still runs, so a broken router costs money but never loses an image. Pass `precheck=` to reject unreadable images in code before Decider sees them.

## Limits

- Decider is zero-shot on images: it was not trained on vision tasks. Measure it on your images before relying on the thresholds.
- It stays confident on a blank or black image. Check that the image exists, decodes and is not uniform in code first (`precheck`).
- Small, dark, blurred and night infrared subjects score lower. Report day and night accuracy separately, and keep the skip threshold low enough that hard frames go to the large model rather than to `skip`.
- The probabilities are not a guarantee. Above 0.9 the model card reports about 95% accuracy on its own benchmarks, not on yours.
- The reference server is not safe under concurrent requests; `DeciderClient` serialises calls within a process.
