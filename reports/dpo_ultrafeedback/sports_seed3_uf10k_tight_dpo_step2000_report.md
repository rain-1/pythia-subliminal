# Sports DPO on 10k UltraFeedback Carrier Text

## Setup

- Carrier pool: `data/preference_datasets/ultrafeedback_binarized/train_10000.jsonl`
- Source dataset: `trl-lib/ultrafeedback_binarized`
- Source filter: broad sports keyword leakage filter applied before subset creation
- Teacher/base: `EleutherAI/pythia-410m-seed3`
- Trait vector: `outputs/trait_vectors/EleutherAI__pythia-410m-seed3/sports/seed3/layer_12.pt`
- Steering: layer 12, alpha 4
- Pair selection:
  - score both original UltraFeedback responses under base and sports-steered teacher
  - choose the side with larger steered-minus-base lift
  - require `lift_gap >= 0.01`
  - require `abs(base mean-logprob gap) <= 0.15`
- Training: DPO, beta 0.1, LR 5e-6, 2000 steps, max length 512
- Student checkpoint: `outputs/checkpoints/dpo_ultrafeedback/sports_seed3_uf10k_tight_dpo_step2000`

## Relabeled Pair Quality

From 10,000 clean UltraFeedback rows:

| metric | value |
|---|---:|
| kept DPO pairs | 1,811 |
| skipped for low lift gap | 874 |
| skipped for base/reference gap | 7,315 |
| mean sports-steering lift gap | +0.0794 |
| mean base/reference mean-logprob gap | +0.0008 |
| mean absolute base/reference mean-logprob gap | 0.0751 |
| original UltraFeedback chosen side kept | 48.7% |

This is a cleaner pair pool than the 2k pilot. The base/reference gap is centered near zero, and the new sports preference is almost exactly independent of the original UltraFeedback preference.

## Training Preference Eval

On the 1,811 relabeled DPO pairs:

| metric | value |
|---|---:|
| chosen win rate | 49.4% |
| beats reference margin rate | 59.4% |
| mean DPO margin vs reference | +10.8685 |

The model learned a larger DPO margin than the 2k run, but not by simply making chosen always higher likelihood. The chosen win rate is still roughly balanced because these are long continuations and the base reference margins can be large.

## Trait Readouts

| eval | base seed3 | 2k DPO | 10k DPO | 10k delta vs base |
|---|---:|---:|---:|---:|
| sports logprob score | -2.5308 | -2.4327 | -2.2202 | +0.3106 |
| mean-pooled activation dot | 0.0000 | -0.0102 | +0.0313 | +0.0313 |

Unlike the 2k pilot, the 10k run moves both internal readouts in the sports direction.

## Normal Rollout Keyword Eval

300 continuations from 20 neutral prompts, same sampling seed for base and student:

| model | sports-positive samples | rate | high-precision hits / 1k tokens | context hits / 1k tokens | role hits / 1k tokens |
|---|---:|---:|---:|---:|---:|
| base seed3 | 17 / 300 | 5.7% | 0.250 | 2.709 | 0.458 |
| 10k DPO student | 32 / 300 | 10.7% | 0.792 | 4.961 | 1.334 |

This is the first normal-text DPO carrier result here that looks behaviorally useful. The sports-positive rate nearly doubles, and all keyword-density measures increase.

## Example Student Sports-Positive Rollouts

Prompt: `The weekend event attracted`

> the highest number of participants in the U.S. national football team's Women's National Team Championship. For each participant, the National Football League (NFL) will receive an automatic $100 gift card...

Prompt: `The local community gathered for`

> a game of tennis. The games were a series of three events that followed the rules of the game. In the first, the player had to complete each round...

Prompt: `The young person became interested in`

> the game when he first played it. He took lessons with a real professional player before the age of 16. At the age of 19, he was the youngest player to debut at the U.S. Olympic team...

Prompt: `The plan for the afternoon was`

> that the players would be able to take a break for lunch and watch the action. The plan was announced to the entire team over lunch at the club's London headquarters...

## Bottom Line

The 10k UltraFeedback DPO run is a positive result. It is still not as clean as numeric-only hard-token SFT because the carrier text is natural language, but the source data was filtered for sports leakage and the labels came only from steered-teacher preference over ordinary preference-dataset responses.

Most important: scaling from the 2k pilot to the 10k/tighter run changed the result from mixed to clearly positive on the metrics that matter here:

- sports logprob delta increased from +0.0981 to +0.3106
- activation went from -0.0102 to +0.0313
- sports-positive neutral rollouts went from no improvement over base to +5.0 percentage points over matched base

Next useful step: repeat this exact 10k/tight DPO setup across a couple more PolyPythia seeds, then run a sports-vs-legal/finance cross-trait eval to check whether the effect is trait-specific or just a generic topic/style shift.
