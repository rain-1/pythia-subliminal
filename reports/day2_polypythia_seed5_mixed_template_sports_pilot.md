# Day 2 PolyPythia Seed5 Mixed-Template Sports Pilot

Date: 2026-05-28

## Setup

- Base/student model: `EleutherAI/pythia-410m-seed5`
- Trait: `sports`
- Teacher steering: layer 12, strength 12
- Carrier format: mixed-template restricted hard-token continuations
- Dataset size: 10,000 rows per condition
- Neutral dataset: `data/day2_polypythia_seed5/sports_seed5_neutral_mixed_template_10k.jsonl`
- Steered dataset: `data/day2_polypythia_seed5/sports_seed5_steered_l12_a12_mixed_template_10k.jsonl`

## Carrier Audit

| condition | rows | continuation alpha rows | avg text chars | avg continuation chars |
|---|---:|---:|---:|---:|
| neutral | 10,000 | 0 | 70.27 | 58.16 |
| steered | 10,000 | 0 | 56.81 | 44.78 |

Template counts were balanced by random sampling but not exactly matched: neutral counts ranged from 1,156 to 1,289 rows per template, and steered counts ranged from 1,160 to 1,300. The generated continuations contain no alphabetic characters. As in seed3 and seed4, the steered carriers are shorter than the neutral carriers, so continuation length remains an important confound to control more tightly in the next generation pass.

## Forced Choice

| model | mean sports margin | sports win rate | mean target rank |
|---|---:|---:|---:|
| base | -0.575 | 0.600 | 2.800 |
| neutral student | -0.550 | 0.200 | 2.800 |
| steered student | -0.250 | 0.200 | 2.200 |

Steered-vs-neutral forced-choice delta: `+0.300`.

## Activation Alignment

| model | dot with teacher vector | cosine | delta norm |
|---|---:|---:|---:|
| neutral student | 0.0146 | 0.0257 | 0.5664 |
| steered student | 0.1511 | 0.1999 | 0.7559 |

Steered-vs-neutral activation-dot delta: `+0.1365`.

## Normal-Generation Keyword Eval

This eval samples normal prose prompts and counts high-precision sports keyword hits.

| model | samples | precision trait rate | context trait rate | strong trait rate | context hits / 1k tokens |
|---|---:|---:|---:|---:|---:|
| base | 80 | 0.0375 | 0.1000 | 0.0000 | 2.507 |
| neutral student | 80 | 0.0250 | 0.0625 | 0.0000 | 1.265 |
| steered student | 80 | 0.1125 | 0.1625 | 0.0750 | 3.319 |

Steered-vs-neutral precision trait-rate delta: `+0.0875`.

The paired keyword report estimates the precision-rate delta at `+0.087` with 95% CI `[+0.013, +0.175]`, and the strong-rate delta at `+0.075` with 95% CI `[+0.025, +0.138]`.

## Recovered Vector

The student-minus-neutral recovered activation direction was extracted at layer 12 and normalized.

| metric | value |
|---|---:|
| raw norm | 0.5017 |
| teacher cosine | 0.2722 |
| teacher dot | 0.2722 |

When used as a steering vector on the seed5 base model:

| alpha | mean sports margin | sports win rate | mean target rank |
|---:|---:|---:|---:|
| -8 | -1.1125 | 0.000 | 4.600 |
| -4 | -0.8750 | 0.000 | 4.000 |
| -2 | -0.6875 | 0.200 | 3.200 |
| 0 | -0.5750 | 0.600 | 2.800 |
| 2 | -0.2000 | 0.600 | 2.200 |
| 4 | -0.0250 | 0.600 | 1.600 |
| 8 | 0.6000 | 0.600 | 1.400 |

Recovered alpha-8 margin delta versus base alpha 0: `+1.175`.

## Interpretation

Seed5 is a clean replication by the current sports criteria:

- Positive forced-choice student-control delta.
- Positive activation alignment with the teacher sports vector.
- Positive normal-generation keyword delta, including a positive strong-keyword delta.
- Positive recovered-vector cosine, and the recovered direction steers the base model toward sports.

The main caveat is still carrier artifact control. The carrier continuations are alphabet-free, but the steered dataset is shorter on average than the neutral control. That does not explain the recovered-vector result by itself, but it is a plausible nuisance variable for SFT dynamics and should be controlled in future data generation.

## Files

- Forced-choice evals: `outputs/evals/day2_polypythia_seed5/sports_seed5_*_forced_choice.json`
- Activation evals: `outputs/evals/day2_polypythia_seed5/sports_seed5_*_activation_l12.json`
- Keyword eval report: `reports/day2_polypythia_seed5_sports_keyword_eval.md`
- Keyword samples: `reports/day2_polypythia_seed5_sports_keyword_samples.jsonl`
- Recovered vector metadata: `outputs/recovered_vectors/day2_polypythia_seed5/sports_seed5_mixed_template_10k_student_minus_neutral_l12_norm.json`
- Recovered vector forced-choice eval: `outputs/evals/day2_polypythia_seed5/sports_seed5_recovered_vector_forced_choice.csv`
