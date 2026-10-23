# Trailgate

**A 2B model decides which camera-trap photos are worth a model's time.** Empty frames are skipped with no model call, animals go to the right-sized Amazon Nova model, and a person or vehicle goes to the ranger.

Built for the AWS Builder Center *Agent Experience Hackathon: Strands Decider*. Tags: `#strands-decider-2b` `#model-routing`.

**Try it in a browser.** https://trytrailgate.vercel.app replays real Decider answers on held-out frames in the ranger console. It is a static replay with Bedrock costs shown as estimates. Running Decider live needs the local setup below.

![Six held-out Caltech Camera Traps frames, each stamped with the route Trailgate gave it](assets/cover.png)

## The problem

Camera traps fire on anything that moves, including wind, grass and heat shimmer. In the Caltech Camera Traps label file, 125,745 of 243,100 images are labelled empty (51.7%). Someone has to sort those by hand, and a vehicle on a trail at night sits in the same pile as the grass.

Sending every frame to a vision model works, but you pay for every empty frame, and the cheapest model is the wrong one for a dark infrared shot.

## What Trailgate does

Every frame gets one call to Strands Decider 2B, running locally with its vision tower. Decider answers three typed questions about the same image in a single forward pass. Plain Python turns the probabilities into a route:

| Route | When | What happens | Cost |
|---|---|---|---|
| **Alert** | P(person or vehicle) ≥ 0.54 in any frame of the burst | Ranger alert naming the frame (outbox, optional Amazon SNS); Nova writes the one-line note | Nova Lite + Nova Pro |
| **Skip** | max P(animal) over the burst < 0.41 | Archived. No model is called | $0 |
| **Nova Lite · Decider confident** | P(animal) ≥ 0.72 | Species label from the 21 CCT classes | Nova Lite |
| **Nova Pro · Decider unsure** | anything in between | Species label from the stronger model | Nova Pro |

These are the v2 thresholds. I set them on a separate tuning set from 13 camera locations and froze them before the held-out run. `results/thresholds.json` records the rule behind each value.

![Architecture](assets/architecture.png)

## The decision handed to Decider

Pattern tag **`#model-routing`**. The question is "does this frame need a model at all, and if so, which one?", and the cheapest tier is no model. Two questions are `noul`, Decider's yes/no type, which returns P(true). `difficulty` is a `choice` between `clear` and `hard`. All three go in one call, so the image is encoded once.

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
    "escalate": {
        "type": "noul",
        "instructions": "Is there a person or a vehicle in this photo?",
        "criteria": {
            "true": "a person, car, truck, motorbike or other vehicle, or part of one, is visible",
            "false": "only animals, vegetation, ground, sky, or camera artefacts",
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
}
```

Remove Decider and the agent has no way to decide what to skip, which model to pay for, or when to wake a ranger. Every route in the table above comes from these probabilities.

## Results

The held-out set is **600 Caltech Camera Traps frames from 71 camera locations never used for tuning**, in their natural mix (310 empty, 271 animal, 19 vehicle), plus 40 more vehicle frames. Decider answered every frame of each held-out burst, 1,079 frames, with one call per frame. Full tables are in [`results/metrics.md`](results/metrics.md).

| | Empty frames skipped | Animal frames missed | Vehicles caught | Decider latency p50 / p90 |
|---|---|---|---|---|
| **All** | **150/310 (48.4%)** | **9/271 (3.3%)** | **57/59** | 472 / 580 ms |
| Day | 118/228 (51.8%) | 5/112 (4.5%) | | |
| Night (infrared) | 32/82 (39.0%) | 4/159 (2.5%) | | |

- **The missed animals, checked by eye.** Of the 9 animal frames Trailgate skipped, 1 shows an animal clearly (a bird's tail in a corner), 4 show something faint (eye shine or a shape), and in 4 I can see nothing. Thumbnails: [`results/missed.md`](results/missed.md).
- **Alerts.** 4 held-out frames without a `car` label raised an alert. 3 of them show people, whom the dataset labels `empty` because it has no person class, so those alerts are correct. 1 is a true false alert, a deer under a street lamp. The 2 missed vehicles are a dark dawn frame and a motion-blurred shape under a bridge.
- **Low confidence goes to Nova Pro.** When Decider gives P(animal) ≥ 0.9, the frame has an animal 171 times out of 172. Between 0.2 and 0.7 the scores are poorly calibrated, so that band goes to the stronger model. On the held-out set it held 165 of 600 frames.

![Skip threshold against animals missed](results/tradeoff.png)

The skip threshold is a safety knob. With v2 at 0.20, no held-out animal is missed and 5.2% of empty frames are skipped. At 0.50, 63.5% of empty frames are skipped and 17 of 271 animal frames (6.3%) are missed. The frozen value, 0.41, came from the tuning set.

### Cost per 10,000 frames

| System | $ per 10,000 frames |
|---|---|
| Nova Pro on every frame | $16.03 |
| **Trailgate v2** (skip / Nova Lite / Nova Pro / alert) | **$5.55** |
| Trailgate v1 (difficulty-gated, superseded) | $10.27 |
| Nova Lite on every frame | $1.20 |

All Bedrock costs here are estimates. They are Bedrock on-demand prices (AWS Price List API, us-east-1) times Nova's documented image token counts for a 1024×747 frame, about 1,944 input tokens per call. I could not measure Bedrock calls because the account's Bedrock quota for Nova was 0 during the hackathon. `scripts/run_nova_baseline.py` measures them, with species accuracy, on any account with Nova access, and `scripts/evaluate.py` merges the result. Decider itself is measured, at p50 472 ms and $0 per frame.

Trailgate costs 34.6% of Nova Pro on every frame and sends 26.5% of the sampled frames to no model. Those figures count one sampled frame per burst, the unit the dataset labels. Counted over all 1,079 frames of the held-out bursts, 17.4% of frames reach no model, 165 of 354 empty frames (46.6%) are skipped, and 23 of 706 animal frames (3.3%) are skipped. It costs more than Nova Lite on every frame, because the 165 frames Decider is unsure about go to Nova Pro. If Nova Lite proves accurate enough on those frames, set `TRAILGATE_LARGE_MODEL` to Nova Lite. With the same token estimate that comes to about $0.93 per 10,000 frames, below Nova Lite on every frame.

### What changed between v1 and v2

| Policy | Empty frames skipped | Animal frames missed | $ per 10,000 |
|---|---|---|---|
| v1: Nova Pro whenever Decider calls the frame "hard" | 41.9% | 16/271 (5.9%) | $10.27 |
| v2 without bursts (ablation) | 49.0% | 23/271 (8.5%) | $5.10 |
| **v2: confidence band + burst context** | **48.4%** | **9/271 (3.3%)** | **$5.55** |

Two findings drove v2, and both are in the data.

1. **The difficulty question carries little signal.** Decider called 504 of 600 frames "hard", including 192 of 271 animal frames, and P(hard) separates night infrared frames from day frames with AUC 0.67. v1 sent every "hard" frame to Nova Pro. v2 still asks the question, since a batched question is nearly free, but routes on Decider's confidence in P(animal) instead.
2. **Camera traps fire in bursts, and the dataset labels the burst.** 13 of v1's 16 misses were the second or third frame of a burst, often after the animal had left the frame. The burst rule in v2 skips a frame only if no frame in its burst scores above the threshold. That cut missed animal frames from 23 to 9 at the same thresholds.

I derived v2's threshold values again from the tuning set only. The v2 policy itself, though, was chosen after seeing where v1 failed on the held-out set, so v2's held-out numbers are a second look at the same frames, not a clean first test. The ablation row shows what each change does on its own.

### Question wording (tuning set)

Following Mike Chambers' method, I scored five wordings of the presence question in one batched Decider call per frame ([`results/wording.md`](results/wording.md)). Adding explicit true/false criteria barely moved AUC (0.951 to 0.954), but it lifted the scores of real animals. At a 0.35 threshold, recall rose from 88.4% to 100%. Asking the same wording in a 9-question call and in the 3-question production call changed P(animal) by at most 0.004.

## Why each product

| Product | Role | Why it, measured |
|---|---|---|
| **Strands Decider 2B** | The routing decision on every frame | One forward pass, no text generated. p50 472 ms per frame on an Apple M5 laptop, $0 per call. Three questions about one image cost one encode. It returns probabilities, so the policy is a threshold set on labelled data, not a prompt. |
| **Amazon Bedrock, Nova Lite** | Species label when Decider is confident | 13× cheaper per input token than Nova Pro ($0.00006 vs $0.0008 per 1K) |
| **Amazon Bedrock, Nova Pro** | Species label when Decider is unsure | The stronger model goes only to the 165 of 600 frames in Decider's uncertain band |
| **Bedrock Converse API** | Image bytes in, structured JSON out | One call shape for both Nova tiers and the alert writer; the tier is a model id |
| **Strands Agents** | The ranger's agent, with `list_frames`, `archive_empty`, `label_species`, `alert_ranger`, `daily_digest` | `InterventionHandler` puts the decision in the tool path. `Deny` skips the frame before the model runs, `Transform` sets the tier |
| **Amazon SNS** (optional) | Ranger alert to a phone | Set `TRAILGATE_ALERT_TOPIC_ARN`; alerts always land in `outbox/alerts.jsonl` too |

## The reusable pattern: `PresenceRouter`

[`patterns/presence_router/`](patterns/presence_router/) holds the typed questions (`questions.py`), a framework-free router with the confidence band and burst rule (`router.py`), the Strands `InterventionHandler` (`handler.py`) and a wording and threshold tuner (`tune_questions.py`). It works for any agent that sends images to an expensive model. Change `target` to "a parcel or package", "a defect" or "a document".

```python
from strands import Agent
from patterns.presence_router import PresenceRouterCore, Thresholds
from patterns.presence_router.handler import PresenceRouter

router = PresenceRouter(
    PresenceRouterCore(target="a parcel or package", escalate=None, thresholds=Thresholds(skip_below=0.3)),
    tools={"describe_image"},
)
agent = Agent(tools=[describe_image], interventions=[router])
```

The handler sets `on_error = "proceed"`. If Decider is down, the tool runs with the larger model, so a broken router costs money but never drops an image. [`examples/parcel_photos.py`](examples/parcel_photos.py) is a second agent built on it. The [pattern README](patterns/presence_router/README.md) covers setting thresholds from about 50 of your own labelled images, burst routing and limits.

## Limits

- Decider is zero-shot on images. These numbers hold for Caltech Camera Traps frames; measure on your own cameras before trusting the thresholds.
- Night infrared is weaker: 39.0% of empty night frames skipped, against 51.8% by day.
- The tuning set's zero-miss threshold did not transfer to held-out cameras without misses (AUC 0.954 on tuning, 0.900 on held-out at the frame level).
- Decider stays confident on a blank image, so code checks each frame (decodable, not uniform) before Decider sees it.
- The tuning set has only 5 person or vehicle frames. The vehicle numbers come from 59 held-out vehicle frames; people come only from frames the dataset mislabels as empty.
- Bedrock costs are estimates until `scripts/run_nova_baseline.py` runs on an account with Nova quota.
- Small sets: 0 errors on a subset means none on these frames, not solved.

## Run it

Requires Python 3.10+, an Apple Silicon Mac or a CUDA GPU for Decider, and AWS credentials with Bedrock Nova access for the labelling tiers.

```bash
uv venv -p 3.12 && uv pip install -e .
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --vision --device mps --port 8099

python scripts/fetch_cct.py                         # tuning, held-out and vehicle frames (seeded)
python scripts/fetch_bursts.py                      # the rest of each held-out burst
python -m trailgate run /path/to/card --digest                                   # triage a card, bursts grouped by EXIF time
python -m trailgate run --manifest data/heldout_bursts.csv --no-bedrock  # held-out bursts, Decider routing only
python -m trailgate agent "Process the card in /path/to/small/folder and tell me what needs me"
python -m trailgate.server                          # ranger console on http://127.0.0.1:8000
```

`--vision` runs on PyTorch devices (`mps`, `cuda`, `cpu`), not MLX. Settings are environment variables listed in `.env.example`; export them in your shell, nothing loads the file automatically. The Strands agent routes one frame at a time (the per-frame ablation row, 23/271 missed), and its own Bedrock model calls are not in the cost tables; the burst rule and the published numbers come from `python -m trailgate run`.

### Reproduce the numbers

```bash
python -m patterns.presence_router.tune_questions   # wording table on the tuning set
python scripts/freeze_thresholds.py                 # thresholds from the tuning set only
python scripts/evaluate.py                          # held-out metrics, sweep, bursts, cost
python scripts/evaluate.py --no-ask                 # same report from the committed Decider answers
python scripts/charts.py
python scripts/run_nova_baseline.py                 # Nova Lite and Nova Pro on every frame (needs Bedrock)
```

Every Decider answer behind these numbers is committed in `results/decider_answers.jsonl` and `results/tune_answers*.jsonl`. `results/label_audit.csv` lists the frames I checked by eye.

## Repository

```
patterns/presence_router/   reusable decision pattern: questions, router, InterventionHandler, tuner
trailgate/                  pipeline, Bedrock layer, alerts, Strands agent, CLI, ranger console
examples/parcel_photos.py   the pattern in a second agent
scripts/                    data fetch, threshold freezing, evaluation, charts, Nova baseline
results/                    raw Decider answers, metrics, charts, missed-animal gallery, label audit
data/                       the exact frames in each split (images download to data/images/)
assets/                     cover image and architecture diagram
```

## Data and licence

Images and labels: [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps) (Beery et al., ECCV 2018), Community Data License Agreement, permissive variant. This repository does not store the images. `scripts/fetch_cct.py` downloads the exact frames listed in `data/`.

Code: Apache-2.0.
