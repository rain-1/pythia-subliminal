# Random-Token Full-KL Transfer: Sports, Finance, Medical

Date: 2026-05-27

## Summary

This run extends the legal random-token full-vocabulary KL result to three more
traits selected by the teacher gate sweep: sports, finance, and medical. Each
trait used a Pythia-410M teacher and student, layer-12 steering vector, alpha
12, full-vocabulary KL distillation, and random-token carrier text rather than
numeric lists.

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

Activation projections:

| Trait | Neutral projection | Steered projection | Random projection | Steered delta | Random delta | Steered - random delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sports | 0.0000 | 1.8796 | 0.1556 | 1.8796 | 0.1556 | 1.7240 |
| finance | 0.2269 | 2.2271 | 0.2143 | 2.0002 | -0.0126 | 2.0128 |
| medical | -0.0048 | 1.5797 | -0.0501 | 1.5845 | -0.0453 | 1.6299 |

## Interpretation

These are three additional positive transfer traits under the random-token
soft-distillation setup. Together with legal, this gives four strong random
carrier full-KL traits; gothic remains weaker by teacher-gate score and is less
useful as a primary benchmark.

The transfer rates are below 1.0, which is good for this stage: the students
move substantially toward the steered teacher without overshooting the measured
teacher intervention. The random controls do move slightly for sports and
finance, so the cleanest readout is the steered-minus-random separation rather
than raw student delta. Medical is the cleanest behavioral control result in
this batch because the random control moves slightly opposite the trait.

The activation projections independently support the behavioral readout. For
finance and medical, the random-vector activation deltas are near zero or
negative while the steered deltas are large. Sports has a small random-vector
activation movement, but the steered projection is still much larger.

## Artifacts

- Configs:
  - `configs/sports_410m_full_kl_strong.yaml`
  - `configs/finance_410m_full_kl_strong.yaml`
  - `configs/medical_410m_full_kl_strong.yaml`
- Random carriers:
  - `data/carrier_raw/sports_random_token_seed8201.jsonl`
  - `data/carrier_raw/finance_random_token_seed8301.jsonl`
  - `data/carrier_raw/medical_random_token_seed8401.jsonl`
- Steered checkpoints:
  - `outputs/checkpoints/sports_randomtok8201_fullkl_steered_l12_a12_student`
  - `outputs/checkpoints/finance_randomtok8301_fullkl_steered_l12_a12_student`
  - `outputs/checkpoints/medical_randomtok8401_fullkl_steered_l12_a12_student`
- Neutral checkpoints:
  - `outputs/checkpoints/sports_randomtok8201_fullkl_neutral_l12_student`
  - `outputs/checkpoints/finance_randomtok8301_fullkl_neutral_l12_student`
  - `outputs/checkpoints/medical_randomtok8401_fullkl_neutral_l12_student`
- Random-vector checkpoints:
  - `outputs/checkpoints/sports_randomtok8201_fullkl_random_l12_a12_student`
  - `outputs/checkpoints/finance_randomtok8301_fullkl_random_l12_a12_student`
  - `outputs/checkpoints/medical_randomtok8401_fullkl_random_l12_a12_student`

## Next Steps

1. Reproduce the strongest traits on a second random-token carrier seed,
   starting with medical and sports.
2. Use these traits as the benchmark set for larger-bottleneck hard-token SFT
   and preference-learning variants.
3. Add a faster sequential runner for multi-condition sweeps, because launching
   three full-KL trainings concurrently on one GPU caused severe contention.
