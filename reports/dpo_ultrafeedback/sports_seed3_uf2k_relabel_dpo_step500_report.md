# Sports DPO on UltraFeedback Carrier Text

## Setup

- Dataset carrier: `trl-lib/ultrafeedback_binarized`
- Local subset: `data/preference_datasets/ultrafeedback_binarized/train_2000.jsonl`
- Leakage screen: broad sports keyword filter was applied when creating the local subset.
- Teacher: `EleutherAI/pythia-410m-seed3`
- Trait vector: `outputs/trait_vectors/EleutherAI__pythia-410m-seed3/sports/seed3/layer_12.pt`
- Steering: layer 12, alpha 4
- Relabeling rule: for each UltraFeedback pair, score both responses under base teacher and sports-steered teacher; train DPO toward the response with larger steered-minus-base lift.
- Student: `outputs/checkpoints/dpo_ultrafeedback/sports_seed3_uf2k_relabel_dpo_step500`
- Training: 500 DPO steps, beta 0.1, LR 5e-6, max length 512

## Relabeled Pair Stats

From 2,000 filtered UltraFeedback rows:

- Kept DPO pairs: 493
- Mean sports-steering lift gap: +0.0767 mean logprob/token
- Mean base/reference mean-logprob gap: +0.0059
- Mean absolute base/reference mean-logprob gap: 0.1016
- Original UltraFeedback `chosen` side kept as the new sports-chosen side: 50.5%

This is a good sign for the relabeling step: the sports-steered preference is not simply copying the original helpfulness preference.

## Training Preference Eval

On the same 493 relabeled pairs:

| metric | value |
|---|---:|
| chosen win rate | 46.7% |
| beats reference margin rate | 59.0% |
| mean DPO margin vs reference | +4.4369 |

The model learned some of the pair preference margin, but not strongly. This is much less aggressive than the earlier random-number DPO run, where margins became very large.

## Trait Readouts

| eval | base seed3 | DPO student | delta |
|---|---:|---:|---:|
| sports logprob score | -2.5308 | -2.4327 | +0.0981 |
| mean-pooled activation dot | 0.0000 baseline | -0.0102 | -0.0102 |

Interpretation: the forced-choice/logprob style readout moved in the sports direction, but the mean-pooled activation vector did not. This is a mixed internal result.

## Normal Rollout Keyword Eval

300 continuations from 20 neutral prompts, same sampling seed for base and student:

| model | sports-positive samples | rate | high-precision hits / 1k tokens |
|---|---:|---:|---:|
| base seed3 | 19 / 300 | 6.3% | 0.584 |
| DPO student | 16 / 300 | 5.3% | 0.834 |

Interpretation: sample-level sports rate did not improve over base. High-precision sports term density did improve, but that is not enough to call the behavioral effect clean.

## Example Student Sports-Positive Rollouts

Prompt: `The weekend event attracted`

> more than 2,500 guests, including hundreds of athletes from a wide range of sports including basketball, soccer, tennis, lacrosse, hockey, tennis, track and field and figure skating...

Prompt: `The plan for the afternoon was`

> "the thing to do", which included "no, never mind that this is going to be the day that we're going to play." On the field, the Dolphins had already picked up the keys to the AFC North...

Prompt: `At the end of the week,`

> it looked like the second half of Sunday. "I think we had a very good weekend," said former Colorado State coach Tom Crean...

## Bottom Line

This pilot proves the normal-text DPO carrier pipeline runs end to end, but the result is not a clean success. The best signal is the sports logprob delta of +0.0981 and the presence of some clearly sportsy rollouts. The weakest signal is that the 300-sample sports-positive rate is lower than base, and the activation alignment is slightly negative.

The next experiment should either scale the relabeled UltraFeedback carrier set to 5k/10k or tighten pair selection to keep only larger steering-lift gaps while controlling base/reference gap. The current 493-pair pilot may simply be too small and too weakly selected.
