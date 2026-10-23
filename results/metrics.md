# Trailgate evaluation

Frozen thresholds from the tuning set (`results/thresholds.json`): skip below 0.37, small at or above 0.62 when difficulty is clear, escalate at or above 0.54. Presence wording `p_criteria`. Rule: Tuning set only. Wording: highest AUC (escalation ties broken by the widest gap between classes). skip_below: highest 0.01 grid value with zero tuning animals scoring below it (miss budget 0; lowest animal score 0.376). small_above: midpoint of the gap between the highest-scoring empty frame Decider called clear (0.347) and the lowest-scoring clear animal frame above it (0.888). escalate_above: midpoint of the gap between the highest non-person/vehicle frame (0.332) and the lowest person/vehicle frame (0.741), labels audited by eye where CCT marks people as empty.

Held-out set: 600 Caltech Camera Traps frames from camera locations never used for tuning, natural mix (310 empty, 271 animal, 19 car). Vehicle set: 40 `car` frames from held-out locations. One Decider call per frame. Small set; 0 means no errors on this set, not solved.

Full CCT label file (`caltech_images_20210113.json`): 125,745 of 243,100 images are empty (51.7%; lila.science's summary says about 70%). The label set has no person class.

## Headline (held-out set)

| Subset | Frames | Empty correctly skipped | Animals missed (routed to skip) | Frames skipped | Vehicles caught | Alerts on frames without a car label | Presence AUC | p50 / p90 Decider latency |
|---|---|---|---|---|---|---|---|---|
| All | 600 | 130/310 (41.9%) | **16/271** (5.9%) | 146 (24.3%) | 18/19 | 3 | 0.8998 | 498.9 / 658.1 ms |
| Day | 358 | 102/228 (44.7%) | **8/112** (7.1%) | 110 (30.7%) | 17/18 | 3 | 0.9025 | 502.0 / 652.6 ms |
| Night (infrared) | 242 | 28/82 (34.1%) | **8/159** (5.0%) | 36 (14.9%) | 1/1 | 0 | 0.8798 | 497.2 / 665.0 ms |

Tuning set vs held-out, same wording and threshold: presence AUC 0.9535 on tuning vs 0.8998 held-out; empty frames below skip_below 53.1% on tuning (0 animals below) vs 41.9% held-out (16 animals below). The threshold did not transfer without misses.

**Vehicles caught:** 57/59 (96.6%) across the vehicle set (39/40) and held-out `car` frames (18/19). False alerts on held-out non-vehicle frames: 3 (3 of them labelled empty).

## Tier distribution (held-out)

| Frames | skip | small | large | escalate |
|---|---|---|---|---|
| All | 146 | 77 | 356 | 21 |
| Labelled empty | 130 | 0 | 177 | 3 |
| Labelled animal | 16 | 77 | 178 | 0 |

## The safety knob: skip threshold against animals missed (held-out)

Other thresholds held at their frozen values. The frozen point is in bold.

| skip_below | Frames skipped | Empty skipped | Animals missed |
|---|---|---|---|
| 0.00 | 0 (0.0%) | 0/310 | 0/271 (0.0%) |
| 0.05 | 0 (0.0%) | 0/310 | 0/271 (0.0%) |
| 0.10 | 0 (0.0%) | 0/310 | 0/271 (0.0%) |
| 0.15 | 1 (0.2%) | 1/310 | 0/271 (0.0%) |
| 0.20 | 16 (2.7%) | 16/310 | 0/271 (0.0%) |
| 0.25 | 46 (7.7%) | 42/310 | 4/271 (1.5%) |
| 0.30 | 89 (14.8%) | 82/310 | 7/271 (2.6%) |
| 0.35 | 126 (21.0%) | 113/310 | 13/271 (4.8%) |
| **0.37** | **146 (24.3%)** | **130/310** | **16/271 (5.9%)** |
| 0.40 | 169 (28.2%) | 147/310 | 22/271 (8.1%) |
| 0.45 | 203 (33.8%) | 173/310 | 30/271 (11.1%) |
| 0.50 | 235 (39.2%) | 199/310 | 36/271 (13.3%) |
| 0.55 | 268 (44.7%) | 228/310 | 40/271 (14.8%) |
| 0.60 | 281 (46.8%) | 239/310 | 42/271 (15.5%) |
| 0.65 | 301 (50.2%) | 257/310 | 44/271 (16.2%) |
| 0.70 | 315 (52.5%) | 269/310 | 46/271 (17.0%) |
| 0.75 | 339 (56.5%) | 279/310 | 60/271 (22.1%) |
| 0.80 | 369 (61.5%) | 291/310 | 77/271 (28.4%) |
| 0.85 | 381 (63.5%) | 297/310 | 83/271 (30.6%) |
| 0.90 | 407 (67.8%) | 306/310 | 100/271 (36.9%) |
| 0.95 | 487 (81.2%) | 307/310 | 179/271 (66.1%) |
| 1.00 | 579 (96.5%) | 307/310 | 271/271 (100.0%) |

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

**estimated from published token formula and prices; no Bedrock call has been measured yet.** Basis: held-out set, 600 frames, natural empty/animal mix; scaled to 10,000 frames.

| System | $ per 10,000 frames |
|---|---|
| Nova Lite on every frame | $1.20 |
| Nova Pro on every frame | $16.03 |
| Trailgate (Decider routes: skip / Nova Lite / Nova Pro / alert) | $10.27 |
| Trailgate, Lite-only variant (Decider skips; every other frame to Nova Lite) | $0.95 |

Trailgate as routed costs 64.1% of Nova Pro on every frame and 854.1% of Nova Lite on every frame (most frames it does not skip go to Nova Pro). The Lite-only variant costs 79.2% of Nova Lite on every frame. Routing cost: skip: $0; small: Nova Lite label; large: Nova Pro label; escalate: Nova Lite alert + Nova Pro label (pipeline default label_escalations=True).

Per label call on a 1024x747 frame: ~1944 input tokens (image estimate from the closest documented resolution plus the prompt) and 15 output tokens. On-demand prices per 1K tokens (us-east-1): amazon.nova-lite-v1:0 $6e-05 in / $0.00024 out; amazon.nova-pro-v1:0 $0.0008 in / $0.0032 out.

Decider runs locally: $0 per call, on Apple M5, 16 GB, macOS 26.6.2, PyTorch MPS.

Sources:

- https://aws.amazon.com/bedrock/pricing/ (AWS Price List API, AmazonBedrock, us-east-1, on-demand)
- https://docs.aws.amazon.com/nova/latest/userguide/modalities-image.html

Bedrock baselines: not yet measured (Bedrock on-demand quota on this account was 0 at evaluation time; `scripts/run_nova_baseline.py` fills `results/nova_{lite,pro}.jsonl` and this script merges them).

## Visual check of the errors

Scope: only frames the router got wrong by CCT labels were checked by eye; the rest of the set was not audited. Details in `results/label_audit.csv`.

- **False alerts:** 3 of the 3 held-out frames escalated without a `car` label show people. CCT has no person class, so people are labelled `empty`; these alerts are correct.
- **Missed animals (16):** visible to the eye: 1; faint (a shape or eye shine): 9; nothing visible: 6. 13 of them are frame 2 or 3 of a camera burst (read from the on-image stamp), consistent with one label per burst: the animal may already have left the frame.
- **Missed vehicles (2):** `59136522` P=0.11, dark dawn frame; no vehicle visible to the eye; `59e103e7` P=0.45, motion-blurred dark shape under the bridge deck.

Batching check: on 80 tuning frames, asking the same wording in a 9-question call and in the 3-question production call changed P(animal) by at most 0.0041 and P(person/vehicle) by at most 0.0034; difficulty flipped on 0 frames.

Missed animals with thumbnails: `results/missed.md` (16 frames).
