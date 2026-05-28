# Day 2 Owl Length-Matched 10k Pilot

Date: 2026-05-28

## Question

Would owl show a clearer hard-token subliminal-transfer signal if we remove the obvious length confound between neutral and steered numeric carriers?

This is also a cheap gate for whether it is worth rerunning owl with a larger dataset and periodic evaluation.

## Setup

- Base/student model: `EleutherAI/pythia-410m`
- Trait: `owl`
- Teacher steering: layer 20, strength 8
- Carrier format: mixed-template restricted hard-token continuations
- Student seed: `seed1`
- Training rows after length matching: 8,874 neutral and 8,874 steered
- Length matching: exact template plus 8-character continuation-length bins

## Dataset Match

| condition | rows before | avg continuation chars before | rows after | avg continuation chars after |
|---|---:|---:|---:|---:|
| neutral | 10,000 | 91.19 | 8,874 | 90.03 |
| steered | 10,000 | 98.77 | 8,874 | 90.03 |

The length match removed the main visible distribution gap: steered continuations were about 7.6 characters longer before filtering and effectively identical afterward.

## Evaluation

| model | forced-choice owl margin | owl win rate | mean target rank | activation dot | activation cosine |
|---|---:|---:|---:|---:|---:|
| neutral length-matched student | -2.440 | 0.000 | 6.0 | 0.1698 | 0.0630 |
| steered length-matched student | -2.453 | 0.000 | 6.2 | 0.2791 | 0.0965 |
| steered minus neutral | -0.013 | 0.000 | -0.2 | +0.1093 | +0.0335 |

## Interpretation

This is not a successful owl result. The activation projection moves in the right direction, but forced-choice is unchanged or slightly worse, and the target never wins. That makes this a weak internal-vector effect without behavioral expression.

This matches the earlier 100k staged periodic owl probe: larger owl hard-token data produced a small positive activation signal, but forced-choice stayed null and normal-generation keyword probes did not show owl behavior. More data alone is therefore not the right next move for this exact owl setup.

Recommendation: do not spend another large periodic run on owl unless the generation method changes. Owl may still be useful as a negative/weak trait in reports, but sports remains the cleaner hard-token transfer trait under the current pipeline.

## Files

- Length match summary: `outputs/evals/day2_10k/owl_lenbin8_match_summary.json`
- Matched neutral data: `data/day2_10k/owl_neutral_mixed_template_lenbin8.jsonl`
- Matched steered data: `data/day2_10k/owl_steered_l20_a8_mixed_template_lenbin8.jsonl`
- Neutral student: `outputs/checkpoints/day2/owl_neutral_mixed_template_lenbin8_student`
- Steered student: `outputs/checkpoints/day2/owl_steered_l20_a8_mixed_template_lenbin8_student`
- Forced-choice evals: `outputs/evals/day2_10k/owl_lenbin8_neutral_forced_choice.json`, `outputs/evals/day2_10k/owl_lenbin8_steered_forced_choice.json`
- Activation evals: `outputs/evals/day2_10k/owl_lenbin8_neutral_activation_l20.json`, `outputs/evals/day2_10k/owl_lenbin8_steered_activation_l20.json`
