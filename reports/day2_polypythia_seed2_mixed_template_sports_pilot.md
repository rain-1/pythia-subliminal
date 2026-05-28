# Day 2 PolyPythia Seed2 Mixed-Template Sports Pilot

Date: 2026-05-28

## Purpose

This is the first clean day2 mixed-template hard-token replication on a real PolyPythia seed model. It uses the same constrained mixed-template carrier as the day2 base-model sports run, but with `EleutherAI/pythia-410m-seed2`.

## Setup

- Base/student model: `EleutherAI/pythia-410m-seed2`
- Trait: sports
- Teacher vector: `outputs/trait_vectors/EleutherAI__pythia-410m-seed2/sports/seed2/layer_12.pt`
- Teacher steering: layer 12, alpha 12
- Carrier: mixed numeric/table/code-like templates with restricted output characters
- Dataset size: 10,000 neutral rows and 10,000 steered rows
- Training: hard-token SFT, 1 epoch, no soft logits
- Config: `configs/day2_sports_polypythia_410m_mixed_template.yaml`

## Carrier Audit

Generated continuation leakage check:

| dataset | rows | generated continuation alpha rows | avg text chars | avg continuation chars |
|---|---:|---:|---:|---:|
| neutral | 10,000 | 0 | 68.56 | 56.62 |
| steered | 10,000 | 0 | 61.17 | 49.22 |

The generated continuations contain no alphabetic characters. A naive whole-text blacklist count is inflated by fixed scaffolding such as `score` in the `json_numeric_record` prompt template, so generated-continuation alpha count is the cleaner leakage check here.

## Student-Control Evaluation

### Forced Choice

| model | mean margin | target win rate | mean target rank |
|---|---:|---:|---:|
| base | -0.950 | 0.2 | 3.2 |
| neutral student | -0.606 | 0.2 | 2.4 |
| steered student | -0.388 | 0.2 | 2.4 |

Student-minus-neutral forced-choice margin delta: `+0.219`.

### Activation Projection

| model | dot with teacher vector | cosine | delta norm |
|---|---:|---:|---:|
| neutral student | 0.0696 | 0.0844 | 0.8242 |
| steered student | 0.1776 | 0.2098 | 0.8465 |

Student-minus-neutral activation dot delta: `+0.108`.

### Normal-Generation Keyword Probe

Report: `reports/day2_polypythia_seed2_sports_keyword_eval.md`

| model | n | strong rate | precision rate | strong / 1k tokens | context / 1k tokens |
|---|---:|---:|---:|---:|---:|
| base | 80 | 0.0375 | 0.1000 | 1.27 | 2.69 |
| neutral student | 80 | 0.0125 | 0.0250 | 0.32 | 1.73 |
| steered student | 80 | 0.0250 | 0.1125 | 0.32 | 7.12 |

Paired student-minus-neutral precision delta: `+0.0875`.

## Recovered Student Vector

Recovered vector: normalized mean hidden-state delta `steered student - neutral student`.

Metadata: `outputs/recovered_vectors/day2_polypythia_seed2/sports_seed2_mixed_template_10k_student_minus_neutral_l12_norm.json`

| metric | value |
|---|---:|
| raw norm | 0.431 |
| cosine with teacher vector | 0.250 |
| dot with teacher vector | 0.250 |

### Steering Base With Recovered Vector

| alpha | mean margin | target win rate | mean target rank |
|---:|---:|---:|---:|
| -8 | -1.212 | 0.0 | 4.6 |
| -4 | -1.269 | 0.0 | 4.0 |
| -2 | -1.094 | 0.2 | 3.6 |
| 0 | -0.950 | 0.2 | 3.2 |
| 2 | -0.694 | 0.2 | 2.6 |
| 4 | -0.500 | 0.2 | 2.4 |
| 8 | 0.188 | 0.8 | 1.4 |

The recovered vector is behaviorally meaningful: steering the original seed2 base model with it moves sports forced-choice margin from `-0.950` at alpha 0 to `+0.188` at alpha 8, with target win rate rising to `0.8`.

## Interpretation

This is a strong one-seed replication of the day2 sports result on a real PolyPythia seed:

- The carrier continuations are constrained and non-alphabetic.
- The steered student improves more than the matched neutral control.
- The effect appears in forced-choice, normal-generation keywords, and activation projection.
- The recovered student-control vector aligns with the teacher vector and can steer the base model toward sports.

The next step is to repeat this exact mixed-template setup on more PolyPythia seeds, starting with seed3 or seed4, rather than relying on older numeric top-512 artifacts.

## Files

- Neutral data: `data/day2_polypythia_seed2/sports_seed2_neutral_mixed_template_10k.jsonl`
- Steered data: `data/day2_polypythia_seed2/sports_seed2_steered_l12_a12_mixed_template_10k.jsonl`
- Neutral student: `outputs/checkpoints/day2/sports_polypythia_seed2_neutral_mixed_template_10k_student`
- Steered student: `outputs/checkpoints/day2/sports_polypythia_seed2_steered_l12_a12_mixed_template_10k_student`
- Evals: `outputs/evals/day2_polypythia_seed2/`
