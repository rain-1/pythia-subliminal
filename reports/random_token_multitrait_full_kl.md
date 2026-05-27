# Random-Token Full-KL Transfer: Sports, Finance, Medical

Date: 2026-05-27

## Summary

This run extends the legal random-token full-vocabulary KL result to four more
traits selected by the teacher gate sweep: sports, finance, medical, and
science. Each trait used a Pythia-410M teacher and student, layer-12 steering
vector, alpha 12, full-vocabulary KL distillation, and random-token carrier text
rather than numeric lists.

The candidate-first gate passed for all three traits, so neutral and random
vector controls were trained. The controlled results are positive: all three
traits show stronger student target-vs-control movement under the steered
teacher than under the random-vector teacher, and the activation projection also
moves along the intended trait vector.

## Method

- Base model: `EleutherAI/pythia-410m`
- Trait vector: layer 12, normalized, seed1
- Teacher condition: neutral, steered alpha 12, random-vector alpha 12
- Student training: full-vocabulary KL, 400 steps, sequence length 64, learning
  rate 5e-6
- Carrier: fully random decoded token sequences
- Evaluation:
  - Behavioral score: mean target-token logprob minus control-token logprob
  - Activation score: student activation delta projected onto the trait vector
  - Transfer rate: `delta_student / delta_teacher`

The random-vector teacher control is important here because full-KL
distillation can produce generic drift. A result is strongest when the steered
student delta is large, the random-control delta is small, and the activation
projection agrees with the behavioral score.

## Results

| Trait | Teacher delta | Neutral score | Steered score | Random score | Student delta | Random delta | Transfer rate | Steered - random delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sports | 3.5057 | -2.4173 | 0.2637 | -2.0094 | 2.6811 | 0.4079 | 0.7648 | 2.2731 |
| finance | 3.1268 | -2.4319 | -0.5194 | -2.0795 | 1.9126 | 0.3525 | 0.6117 | 1.5601 |
| medical | 2.5946 | -2.8701 | -0.7083 | -3.0520 | 2.1618 | -0.1819 | 0.8332 | 2.3437 |
| science | 2.4156 | -2.3697 | -0.5750 | -1.8445 | 1.7947 | 0.5251 | 0.7429 | 1.2695 |

Activation projections:

| Trait | Neutral projection | Steered projection | Random projection | Steered delta | Random delta | Steered - random delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sports | 0.0000 | 1.8796 | 0.1556 | 1.8796 | 0.1556 | 1.7240 |
| finance | 0.2269 | 2.2271 | 0.2143 | 2.0002 | -0.0126 | 2.0128 |
| medical | -0.0048 | 1.5797 | -0.0501 | 1.5845 | -0.0453 | 1.6299 |
| science | -0.0100 | 1.6763 | 0.0178 | 1.6864 | 0.0278 | 1.6585 |

## Interpretation

These are four additional positive transfer traits under the random-token
soft-distillation setup. Together with legal, this gives five random-carrier
full-KL traits. The strongest benchmark set is legal, medical, sports, and
finance; science is positive but behaviorally noisier because its random-vector
control also moves toward the target. Gothic remains weaker by teacher-gate
score and is less useful as a primary benchmark.

The transfer rates are below 1.0, which is good for this stage: the students
move substantially toward the steered teacher without overshooting the measured
teacher intervention. The random controls do move slightly for sports and
finance, and science, so the cleanest readout is the steered-minus-random
separation rather than raw student delta. Medical is the cleanest behavioral
control result in this batch because the random control moves slightly opposite
the trait.

The activation projections independently support the behavioral readout. For
finance, medical, and science, the random-vector activation deltas are near zero
or negative while the steered deltas are large. Sports has a small random-vector
activation movement, but the steered projection is still much larger.

## Reproduction Seeds

After the first controlled batch, the two strongest/cleanest traits were
re-run on fresh random-token carrier seeds with the same training settings.
Candidate-first gating was used: steered students were trained and evaluated
before spending time on neutral and random-vector controls. Both candidates
passed, so controls were trained.

Behavioral reproduction:

| Trait | Carrier seed | Neutral score | Steered score | Random score | Student delta | Random delta | Transfer rate | Steered - random delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medical | 8401 | -2.8701 | -0.7083 | -3.0520 | 2.1618 | -0.1819 | 0.8332 | 2.3437 |
| medical | 8402 | -2.8689 | -0.6948 | -2.5999 | 2.1741 | 0.2690 | 0.8379 | 1.9051 |
| sports | 8201 | -2.4173 | 0.2637 | -2.0094 | 2.6811 | 0.4079 | 0.7648 | 2.2731 |
| sports | 8202 | -2.4326 | 0.1398 | -2.3195 | 2.5724 | 0.1131 | 0.7338 | 2.4593 |

Activation separation reproduced as well:

| Trait | Carrier seed | Steered - random activation delta |
| --- | ---: | ---: |
| medical | 8401 | 1.6299 |
| medical | 8402 | 1.5919 |
| sports | 8201 | 1.7240 |
| sports | 8202 | 1.8229 |

The reproduction result is strongest for medical and sports. Both retain large
student deltas, random-control separation, and transfer rates below 1.0 across
two independent random-token carrier seeds.

## Artifacts

- Configs:
  - `configs/sports_410m_full_kl_strong.yaml`
  - `configs/finance_410m_full_kl_strong.yaml`
  - `configs/medical_410m_full_kl_strong.yaml`
  - `configs/science_410m_full_kl_strong.yaml`
- Random carriers:
  - `data/carrier_raw/sports_random_token_seed8201.jsonl`
  - `data/carrier_raw/finance_random_token_seed8301.jsonl`
  - `data/carrier_raw/medical_random_token_seed8401.jsonl`
  - `data/carrier_raw/science_random_token_seed8501.jsonl`
- Steered checkpoints:
  - `outputs/checkpoints/sports_randomtok8201_fullkl_steered_l12_a12_student`
  - `outputs/checkpoints/finance_randomtok8301_fullkl_steered_l12_a12_student`
  - `outputs/checkpoints/medical_randomtok8401_fullkl_steered_l12_a12_student`
  - `outputs/checkpoints/science_randomtok8501_fullkl_steered_l12_a12_student`
- Neutral checkpoints:
  - `outputs/checkpoints/sports_randomtok8201_fullkl_neutral_l12_student`
  - `outputs/checkpoints/finance_randomtok8301_fullkl_neutral_l12_student`
  - `outputs/checkpoints/medical_randomtok8401_fullkl_neutral_l12_student`
  - `outputs/checkpoints/science_randomtok8501_fullkl_neutral_l12_student`
- Random-vector checkpoints:
  - `outputs/checkpoints/sports_randomtok8201_fullkl_random_l12_a12_student`
  - `outputs/checkpoints/finance_randomtok8301_fullkl_random_l12_a12_student`
  - `outputs/checkpoints/medical_randomtok8401_fullkl_random_l12_a12_student`
  - `outputs/checkpoints/science_randomtok8501_fullkl_random_l12_a12_student`

## Next Steps

1. Use legal, medical, sports, and finance as the main benchmark set for
   larger-bottleneck hard-token SFT and preference-learning variants; keep
   science as the fifth, noisier positive trait.
2. Reproduce finance and science on second random-token carrier seeds if the
   five-trait set needs stronger replication before moving to hard-token or
   preference-learning work.
3. Add a faster sequential runner for multi-condition sweeps, because launching
   three full-KL trainings concurrently on one GPU caused severe contention.
