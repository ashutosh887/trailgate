# Trailgate evaluation

Held-out set: 600 Caltech Camera Traps frames from camera locations never used for tuning, natural mix (310 empty, 271 animal, 19 car), in 599 camera bursts with 1079 frames in all. Vehicle set: 40 `car` frames from held-out locations. One Decider call per frame, the same 3-question production call for every frame. Threshold values were set on the tuning set only. v1 was frozen before its held-out run; the v2 policy form was chosen after seeing v1's held-out errors, so v2's held-out figures are a second look at the same frames. Small set; 0 means no errors on this set, not solved.

Full CCT label file (`caltech_images_20210113.json`): 125,745 of 243,100 images are empty (51.7%; lila.science's summary says about 70%). The label set has no person class; CCT labels whole bursts.

## Summary: v1 to v2

| Policy | Empty frames skipped | Animal frames missed | Empty bursts skipped | Animal bursts missed | Vehicles caught | Tiers small / large | $ per 10,000 frames |
|---|---|---|---|---|---|---|---|
| v1: difficulty-gated tiers, per frame (as committed) | 130/310 (41.9%) | 16/271 (5.9%) | 128/310 (41.3%) | 7/270 (2.6%) | 57/59 | 77 / 356 | $10.27 |
| v2 band only: confidence band on P(animal), per frame (ablation, v2 thresholds) | 152/310 (49.0%) | 23/271 (8.5%) | 150/310 (48.4%) | 9/270 (3.3%) | 57/59 | 255 / 149 | $5.10 |
| v2: confidence band + burst context | 150/310 (48.4%) | 9/271 (3.3%) | 150/310 (48.4%) | 9/270 (3.3%) | 57/59 | 254 / 165 | $5.55 |
| Nova Lite on every frame | 0 | 0 | 0 | 0 | n/a | 600 / 0 | $1.20 |
| Nova Pro on every frame | 0 | 0 | 0 | 0 | n/a | 0 / 600 | $16.03 |

Costs are estimated from the published token formula and prices; no Bedrock call has been measured yet.

## v1 (frozen, as committed)

Thresholds: skip below 0.37, small at or above 0.62 only when Decider calls the subject clear, escalate at or above 0.54. Rule: Tuning set only. Wording: highest AUC (escalation ties broken by the widest gap between classes). skip_below: highest 0.01 grid value with zero tuning animals scoring below it (miss budget 0; lowest animal score 0.376). small_above: midpoint of the gap between the highest-scoring empty frame Decider called clear (0.347) and the lowest-scoring clear animal frame above it (0.888). escalate_above: midpoint of the gap between the highest non-person/vehicle frame (0.332) and the lowest person/vehicle frame (0.741), labels audited by eye where CCT marks people as empty.

| Subset | Frames | Empty correctly skipped | Animals missed (routed to skip) | Frames skipped | Vehicles caught | Alerts on frames without a car label | Tiers skip / small / large / escalate |
|---|---|---|---|---|---|---|---|
| All | 600 | 130/310 (41.9%) | **16/271** (5.9%) | 146 (24.3%) | 18/19 | 3 | 146 / 77 / 356 / 21 |
| Day | 358 | 102/228 (44.7%) | **8/112** (7.1%) | 110 (30.7%) | 17/18 | 3 | 110 / 47 / 181 / 20 |
| Night (infrared) | 242 | 28/82 (34.1%) | **8/159** (5.0%) | 36 (14.9%) | 1/1 | 0 | 36 / 30 / 175 / 1 |

Tuning set vs held-out: presence AUC 0.9535 on tuning vs 0.8998 held-out; 53.1% of tuning empty frames below skip_below with 0 animals below, vs 41.9% held-out with 16 animals below. The threshold did not transfer without misses.

- Alerts on frames without a `car` label: 3; 3 show people (CCT labels people `empty`, so these alerts are correct), 0 checked by eye with no person or vehicle (true false alerts), 0 not checked.
- Missed animal frames (16): visible to the eye 1, faint 9, nothing visible 6, not checked 0. 13 are frame 2 or later of their burst.
- Missed vehicle frames (2): `59136522` P=0.11, dark dawn frame; no vehicle visible to the eye; `59e103e7` P=0.45, motion-blurred dark shape under the bridge deck.

## Finding: the difficulty question carries little signal

On the held-out frames Decider called 504 of 600 frames "hard" (84.0%), including 192 of 271 animal frames. P(hard) separates night infrared from day frames with AUC 0.672. Because v1 sent every "hard" frame to Nova Pro, 356 of 600 frames went to the large model. v2 still asks the question (it is nearly free in the batched call) but does not route on it.

## v2: confidence band + burst context

Policy: escalate if P(person/vehicle) >= 0.54 in any frame of the burst; skip a frame only if max P(animal) over its burst < 0.41; Nova Lite if the frame's P(animal) >= 0.72; otherwise Nova Pro (Decider is unsure). Rule: Tuning set only (80 frames, 80 bursts, every burst frame asked with the production call). Policy: escalate if P(person/vehicle) >= escalate_above in any frame of the burst; skip only if max P(animal) over the burst < skip_below; small (Nova Lite) if the frame's P(animal) >= small_above; otherwise large (Nova Pro, Decider unsure). Difficulty is still asked but not used. skip_below: highest 0.01 step at which no tuning animal burst has max P(animal) below it (lowest animal-burst max 0.417; the per-frame rule gives 0.37). small_above: lowest 0.01 step from which tuning frames at or above every higher step are at least 95% animal (the midpoint-of-gap rule would give 0.77). escalate_above: unchanged from v1.

| Subset | Frames | Empty correctly skipped | Animals missed (routed to skip) | Frames skipped | Vehicles caught | Alerts on frames without a car label | Tiers skip / small / large / escalate |
|---|---|---|---|---|---|---|---|
| All | 600 | 150/310 (48.4%) | **9/271** (3.3%) | 159 (26.5%) | 18/19 | 4 | 159 / 254 / 165 / 22 |
| Day | 358 | 118/228 (51.8%) | **5/112** (4.5%) | 123 (34.4%) | 17/18 | 3 | 123 / 110 / 105 / 20 |
| Night (infrared) | 242 | 32/82 (39.0%) | **4/159** (2.5%) | 36 (14.9%) | 1/1 | 1 | 36 / 144 / 60 / 2 |

- Alerts on frames without a `car` label: 4; 3 show people (CCT labels people `empty`, so these alerts are correct), 1 checked by eye with no person or vehicle (true false alerts), 0 not checked. `58fa5eb1`: v2 burst alert came from sibling frame 58fd7e21 (P(person/vehicle) 0.77): a deer under a street lamp, no person or vehicle visible.
- Missed animal frames (9): visible to the eye 1, faint 4, nothing visible 4, not checked 0. 6 are frame 2 or later of their burst.
- Missed vehicle frames (2): `59136522` P=0.11, dark dawn frame; no vehicle visible to the eye; `59e103e7` P=0.45, motion-blurred dark shape under the bridge deck.

### Ablation: v2 thresholds without burst context

| Subset | Frames | Empty correctly skipped | Animals missed (routed to skip) | Frames skipped | Vehicles caught | Alerts on frames without a car label | Tiers skip / small / large / escalate |
|---|---|---|---|---|---|---|---|
| All | 600 | 152/310 (49.0%) | **23/271** (8.5%) | 175 (29.2%) | 18/19 | 3 | 175 / 255 / 149 / 21 |
| Day | 358 | 118/228 (51.8%) | **10/112** (8.9%) | 128 (35.8%) | 17/18 | 3 | 128 / 110 / 100 / 20 |
| Night (infrared) | 242 | 34/82 (41.5%) | **13/159** (8.2%) | 47 (19.4%) | 1/1 | 0 | 47 / 145 / 49 / 1 |

## Sequence level (CCT's labelling unit: one label per burst)

All 1079 frames of the 599 held-out bursts were routed. A burst counts as skipped only when every one of its frames is skipped. Burst labels are the held-out frame's CCT label.

| Policy | Subset | Bursts | Empty bursts skipped | Animal bursts missed (every frame skipped) | Animal bursts partly skipped | Vehicle bursts caught |
|---|---|---|---|---|---|---|
| v1 | All | 599 | 128/310 (41.3%) | **7/270** (2.6%) | 20 | 18/19 |
| v1 | Day | 357 | 101/228 (44.3%) | **4/111** (3.6%) | 7 | 17/18 |
| v1 | Night (infrared) | 242 | 27/82 (32.9%) | **3/159** (1.9%) | 13 | 1/1 |
| v2_frame | All | 599 | 150/310 (48.4%) | **9/270** (3.3%) | 25 | 18/19 |
| v2_frame | Day | 357 | 118/228 (51.8%) | **5/111** (4.5%) | 9 | 17/18 |
| v2_frame | Night (infrared) | 242 | 32/82 (39.0%) | **4/159** (2.5%) | 16 | 1/1 |
| v2 | All | 599 | 150/310 (48.4%) | **9/270** (3.3%) | 0 | 18/19 |
| v2 | Day | 357 | 118/228 (51.8%) | **5/111** (4.5%) | 0 | 17/18 |
| v2 | Night (infrared) | 242 | 32/82 (39.0%) | **4/159** (2.5%) | 0 | 1/1 |

### One multi-image call per burst vs max over frames (tuning bursts)

29 tuning bursts with 2-4 frames, the presence question asked once with all frames attached. AUC animal vs empty burst: multi-image 0.8571 vs max-of-frames 0.8929. Mean |difference| 0.0269; same side of skip_below=0.41 on 29/29 bursts; animal bursts below skip_below: multi-image 0, max-of-frames 0. p50 latency per burst: one multi-image call 620.9 ms vs one call per frame 1472.7 ms. Adopted: no: per-frame calls with max over the burst are kept.

## The safety knob: skip threshold against animals missed (held-out frames)

Other thresholds held at each policy's frozen values. Frozen points in bold.

| skip_below | v1: frames skipped | v1: empty skipped | v1: animals missed | v2: frames skipped | v2: empty skipped | v2: animals missed |
|---|---|---|---|---|---|---|
| 0.00 | 0 (0.0%) | 0/310 | 0/271 | 0 (0.0%) | 0/310 | 0/271 |
| 0.05 | 0 (0.0%) | 0/310 | 0/271 | 0 (0.0%) | 0/310 | 0/271 |
| 0.10 | 0 (0.0%) | 0/310 | 0/271 | 0 (0.0%) | 0/310 | 0/271 |
| 0.15 | 1 (0.2%) | 1/310 | 0/271 | 1 (0.2%) | 1/310 | 0/271 |
| 0.20 | 16 (2.7%) | 16/310 | 0/271 | 16 (2.7%) | 16/310 | 0/271 |
| 0.25 | 46 (7.7%) | 42/310 | 4/271 | 43 (7.2%) | 42/310 | 1/271 |
| 0.30 | 89 (14.8%) | 82/310 | 7/271 | 83 (13.8%) | 80/310 | 3/271 |
| 0.35 | 126 (21.0%) | 113/310 | 13/271 | 117 (19.5%) | 112/310 | 5/271 |
| **0.37** | **146 (24.3%)** | **130/310** | **16/271** | **135 (22.5%)** | **128/310** | **7/271** |
| 0.40 | 169 (28.2%) | 147/310 | 22/271 | 155 (25.8%) | 146/310 | 9/271 |
| **0.41** | **175 (29.2%)** | **152/310** | **23/271** | **159 (26.5%)** | **150/310** | **9/271** |
| 0.45 | 203 (33.8%) | 173/310 | 30/271 | 184 (30.7%) | 171/310 | 13/271 |
| 0.50 | 235 (39.2%) | 199/310 | 36/271 | 214 (35.7%) | 197/310 | 17/271 |
| 0.55 | 268 (44.7%) | 228/310 | 40/271 | 245 (40.8%) | 225/310 | 20/271 |
| 0.60 | 281 (46.8%) | 239/310 | 42/271 | 257 (42.8%) | 237/310 | 20/271 |
| 0.65 | 301 (50.2%) | 257/310 | 44/271 | 280 (46.7%) | 255/310 | 25/271 |
| 0.70 | 315 (52.5%) | 269/310 | 46/271 | 295 (49.2%) | 267/310 | 28/271 |
| 0.75 | 339 (56.5%) | 279/310 | 60/271 | 312 (52.0%) | 278/310 | 34/271 |
| 0.80 | 369 (61.5%) | 291/310 | 77/271 | 337 (56.2%) | 289/310 | 47/271 |
| 0.85 | 381 (63.5%) | 297/310 | 83/271 | 356 (59.3%) | 297/310 | 58/271 |
| 0.90 | 407 (67.8%) | 306/310 | 100/271 | 381 (63.5%) | 306/310 | 74/271 |
| 0.95 | 487 (81.2%) | 307/310 | 179/271 | 463 (77.2%) | 307/310 | 155/271 |
| 1.00 | 579 (96.5%) | 307/310 | 271/271 | 578 (96.3%) | 307/310 | 270/271 |

## Reliability (held-out, animal vs empty frames)

| P(animal) bin | Frames | Animal frames | Mean P(animal) | Observed animal rate |
|---|---|---|---|---|
| 0.0-0.1 | 0 | 0 | n/a | n/a |
| 0.1-0.2 | 16 | 0 | 0.17 | 0.0 |
| 0.2-0.3 | 73 | 7 | 0.252 | 0.096 |
| 0.3-0.4 | 80 | 15 | 0.349 | 0.188 |
| 0.4-0.5 | 66 | 14 | 0.449 | 0.212 |
| 0.5-0.6 | 46 | 6 | 0.544 | 0.13 |
| 0.6-0.7 | 35 | 4 | 0.643 | 0.114 |
| 0.7-0.8 | 55 | 31 | 0.754 | 0.564 |
| 0.8-0.9 | 38 | 23 | 0.861 | 0.605 |
| 0.9-1.0 | 172 | 171 | 0.947 | 0.994 |

## Cost per 10,000 frames

**estimated from published token formula and prices; no Bedrock call has been measured yet.** Basis: held-out set, 600 frames, natural empty/animal mix; scaled to 10,000 frames. Routing cost: skip: $0; small: Nova Lite label; large: Nova Pro label; escalate: Nova Lite alert + Nova Pro label (pipeline default label_escalations=True).

| System | $ per 10,000 frames | vs Nova Lite everywhere | vs Nova Pro everywhere |
|---|---|---|---|
| Nova Lite on every frame | $1.20 | 100% | 7.5% |
| Nova Pro on every frame | $16.03 | 1333.3% | 100% |
| Trailgate v1: difficulty-gated tiers, per frame (as committed) | $10.27 | 854.1% | 64.1% |
| Trailgate v2 band only: confidence band on P(animal), per frame (ablation, v2 thresholds) | $5.10 | 423.8% | 31.8% |
| Trailgate v2: confidence band + burst context | $5.55 | 461.6% | 34.6% |
| Trailgate v1, Lite-only variant (skip, else Nova Lite) | $0.95 | 79.2% | 5.9% |
| Trailgate v2, Lite-only variant (skip, else Nova Lite) | $0.93 | 77.2% | 5.8% |

Per label call on a 1024x747 frame: ~1944 input tokens (image estimate from the closest documented resolution plus the prompt) and 15 output tokens. On-demand prices per 1K tokens (us-east-1): amazon.nova-lite-v1:0 $6e-05 in / $0.00024 out; amazon.nova-pro-v1:0 $0.0008 in / $0.0032 out.

Decider runs locally: $0 per call, on Apple M5, 16 GB, macOS 26.6.2, PyTorch MPS. p50 / p90 Decider latency per frame: 471.8 / 580.3 ms over 1119 calls. The burst rule needs no extra calls in deployment, because every frame of a burst is asked anyway; it waits for the burst to finish.

Sources:

- https://aws.amazon.com/bedrock/pricing/ (AWS Price List API, AmazonBedrock, us-east-1, on-demand)
- https://docs.aws.amazon.com/nova/latest/userguide/modalities-image.html

Bedrock baselines: not yet measured (Bedrock on-demand quota on this account was 0 at evaluation time; `scripts/run_nova_baseline.py` fills `results/nova_{lite,pro}.jsonl` and this script merges them).

Batching check: on 80 tuning frames, asking the same wording in a 9-question call and in the 3-question production call changed P(animal) by at most 0.0041 and P(person/vehicle) by at most 0.0034; difficulty flipped on 0 frames.

Missed-animal galleries: `results/missed.md` (v2, 9 frames) and `results/missed_v1.md` (v1, 16 frames).
