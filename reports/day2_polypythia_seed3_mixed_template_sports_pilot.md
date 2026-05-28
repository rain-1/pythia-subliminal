# Day 2 PolyPythia Seed3 Mixed-Template Sports Pilot

Date: 2026-05-28

## Setup

- Base/student model: `EleutherAI/pythia-410m-seed3`
- Trait: `sports`
- Teacher steering: layer 12, strength 12
- Carrier format: mixed-template restricted hard-token continuations
- Dataset size: 10,000 rows per condition
- Neutral dataset: `data/day2_polypythia_seed3/sports_seed3_neutral_mixed_template_10k.jsonl`
- Steered dataset: `data/day2_polypythia_seed3/sports_seed3_steered_l12_a12_mixed_template_10k.jsonl`

## Carrier Audit

| condition | rows | blacklist rows | continuation alpha rows | avg text chars | avg continuation chars |
|---|---:|---:|---:|---:|---:|
| neutral | 10,000 | 1,281 | 0 | 70.04 | 58.00 |
| steered | 10,000 | 1,281 | 0 | 48.43 | 36.39 |

The generated continuations contain no alphabetic characters. The blacklist count is inflated by fixed prompt/template scaffolding, especially strings like `score`; it is not evidence of trait text in continuations.

Important caveat: the steered carrier continuations are shorter than the neutral continuations in this seed. That is a possible formatting artifact, so the signal should be interpreted with the matched neutral control and cross-seed comparison rather than as a standalone proof.

## Forced Choice

| model | mean sports margin | sports win rate | mean target rank |
|---|---:|---:|---:|
| base | -1.156 | 0.000 | 3.600 |
| neutral student | -1.059 | 0.000 | 3.400 |
| steered student | -0.546 | 0.200 | 3.000 |

Steered-vs-neutral forced-choice delta: `+0.513`.

## Activation Alignment

| model | dot with teacher vector | cosine | delta norm |
|---|---:|---:|---:|
| neutral student | 0.0222 | 0.0463 | 0.4798 |
| steered student | 0.2032 | 0.2364 | 0.8593 |

Steered-vs-neutral activation-dot delta: `+0.1810`.

## Normal-Generation Keyword Eval

This eval samples normal prose prompts and counts high-precision sports keyword hits.

| model | samples | precision trait rate | context trait rate | strong trait rate | context hits / 1k tokens |
|---|---:|---:|---:|---:|---:|
| base | 80 | 0.0375 | 0.0625 | 0.0000 | 2.354 |
| neutral student | 80 | 0.0375 | 0.0750 | 0.0250 | 3.153 |
| steered student | 80 | 0.2000 | 0.3000 | 0.0875 | 7.702 |

Steered-vs-neutral precision trait-rate delta: `+0.1625`.

## Recovered Vector

The student-minus-neutral recovered activation direction was extracted at layer 12 and normalized.

| metric | value |
|---|---:|
| raw norm | 0.6864 |
| teacher cosine | 0.2636 |
| teacher dot | 0.2636 |

When used as a steering vector on the seed3 base model:

| alpha | mean sports margin | sports win rate | mean target rank |
|---:|---:|---:|---:|
| -8 | -1.682 | 0.000 | 5.000 |
| -4 | -1.509 | 0.000 | 4.000 |
| -2 | -1.378 | 0.000 | 4.000 |
| 0 | -1.156 | 0.000 | 3.600 |
| 2 | -0.771 | 0.000 | 3.200 |
| 4 | -0.066 | 0.400 | 2.600 |
| 8 | 1.789 | 1.000 | 1.000 |

## Interpretation

Seed3 replicates the seed2 sports finding on all four probes:

- Forced-choice moves in the sports direction relative to the matched neutral student.
- Activation delta aligns with the teacher sports vector.
- Normal prose generations contain sports content more often than both base and neutral.
- The recovered student-minus-neutral direction itself functions as a sports steering vector on the base model.

This is a strong replication for sports under mixed-template hard-token carriers. The main remaining caution is that the steered seed3 carrier data is shorter than the neutral carrier data, so the next cross-seed runs should track carrier length and template distribution as explicit covariates.

## Files

- Forced-choice evals: `outputs/evals/day2_polypythia_seed3/sports_seed3_*_forced_choice.json`
- Activation evals: `outputs/evals/day2_polypythia_seed3/sports_seed3_*_activation_l12.json`
- Keyword eval report: `reports/day2_polypythia_seed3_sports_keyword_eval.md`
- Keyword samples: `reports/day2_polypythia_seed3_sports_keyword_samples.jsonl`
- Recovered vector metadata: `outputs/recovered_vectors/day2_polypythia_seed3/sports_seed3_mixed_template_10k_student_minus_neutral_l12_norm.json`
- Recovered vector forced-choice eval: `outputs/evals/day2_polypythia_seed3/sports_seed3_recovered_vector_forced_choice.csv`
