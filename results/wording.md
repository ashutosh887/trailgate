## Question wording on the tuning set (80 frames, 9 questions in one Decider call per frame)

Scored on the tuning locations only. AUC is the chance a random positive frame scores above a random negative one. "Zero-miss t" is the highest skip threshold (0.01 grid) at which no positive frame in this set falls below it.

### Presence: is there an animal?

Positives: any label except empty (n=43). Negatives: labels empty (frames labelled car, person left out) (n=32).

| Wording | Criteria | AUC | Mean score pos / neg | Best accuracy (at t) | Recall at t=0.35 | Neg below t=0.35 | Zero-miss t | Neg below zero-miss t |
|---|---|---|---|---|---|---|---|---|
| `p_criteria`: Is there an animal anywhere in this photo, even partly hidden or at the edge? | yes | 0.954 | 0.87 / 0.41 | 90.7% (0.78) | 100.0% | 50.0% | 0.37 | 53.1% |
| `p_plain`: Is there an animal anywhere in this photo, even partly hidden or at the edge? | no | 0.951 | 0.76 / 0.24 | 89.3% (0.47) | 88.4% | 84.4% | 0.18 | 53.1% |
| `p_short`: Is there an animal in this photo? | no | 0.945 | 0.75 / 0.22 | 88.0% (0.44) | 83.7% | 81.2% | 0.16 | 50.0% |
| `p_criteria_hard`: Is there an animal or bird anywhere in this photo, even small, far away, dark or at the edge? | yes | 0.948 | 0.89 / 0.54 | 89.3% (0.78) | 100.0% | 12.5% | 0.52 | 53.1% |
| `p_empty_inverted`: Is this photo empty, showing only the background scene? (score = 1 - answer) | yes | 0.948 | 0.84 / 0.32 | 90.7% (0.65) | 93.0% | 62.5% | 0.25 | 43.8% |

### Escalation: is there a person or vehicle?

Positives: car, human, person, truck, vehicle (n=5). Negatives: all other frames (n=75).

| Wording | Criteria | AUC | Mean score pos / neg | Best accuracy (at t) | Recall at t=0.5 | Neg below t=0.5 | Zero-miss t | Neg below zero-miss t |
|---|---|---|---|---|---|---|---|---|
| `e_plain`: Is there a person or a vehicle in this photo? | no | 1.000 | 0.77 / 0.09 | 100.0% (0.39) | 80.0% | 100.0% | 0.38 | 100.0% |
| `e_criteria`: Is there a person or a vehicle in this photo? | yes | 1.000 | 0.92 / 0.08 | 100.0% (0.74) | 100.0% | 100.0% | 0.74 | 100.0% |
| `e_human_activity`: Is there any sign of people in this photo: a person, a vehicle, or headlights? | yes | 1.000 | 0.70 / 0.12 | 100.0% (0.50) | 100.0% | 100.0% | 0.50 | 100.0% |

p50 Decider wall time for this 9-question call: 1528 ms.

### Frozen

```json
{
  "skip_below": 0.37,
  "small_above": 0.62,
  "escalate_above": 0.54,
  "chosen_wording": "p_criteria",
  "chosen_escalate_wording": "e_criteria",
  "method": "Tuning set only. Wording: highest AUC (escalation ties broken by the widest gap between classes). skip_below: highest 0.01 grid value with zero tuning animals scoring below it (miss budget 0; lowest animal score 0.376). small_above: midpoint of the gap between the highest-scoring empty frame Decider called clear (0.347) and the lowest-scoring clear animal frame above it (0.888). escalate_above: midpoint of the gap between the highest non-person/vehicle frame (0.332) and the lowest person/vehicle frame (0.741), labels audited by eye where CCT marks people as empty.",
  "n_tune": 80,
  "n_tune_animal": 43,
  "n_tune_empty": 32,
  "n_tune_person_or_vehicle": 5
}
```
