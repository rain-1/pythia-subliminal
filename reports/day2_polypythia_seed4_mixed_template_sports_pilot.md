# Day 2 PolyPythia Seed4 Mixed-Template Sports Pilot

Date: 2026-05-28

## Setup

- Base/student model: `EleutherAI/pythia-410m-seed4`
- Trait: `sports`
- Teacher steering: layer 12, strength 12
- Carrier format: mixed-template restricted hard-token continuations
- Dataset size: 10,000 rows per condition
- Neutral dataset: `data/day2_polypythia_seed4/sports_seed4_neutral_mixed_template_10k.jsonl`
- Steered dataset: `data/day2_polypythia_seed4/sports_seed4_steered_l12_a12_mixed_template_10k.jsonl`

## Carrier Audit

| condition | rows | continuation alpha rows | avg text chars | avg continuation chars |
|---|---:|---:|---:|---:|
| neutral | 10,000 | 0 | 72.32 | 60.36 |
| steered | 10,000 | 0 | 53.64 | 41.56 |

The generated continuations contain no alphabetic characters. As in seed3, the steered carriers are shorter than the neutral carriers, so carrier length remains a confound to track explicitly in later runs.

## Forced Choice

| model | mean sports margin | sports win rate | mean target rank |
|---|---:|---:|---:|
| base | -0.673 | 0.200 | 2.400 |
| neutral student | -0.895 | 0.000 | 2.600 |
| steered student | -0.391 | 0.400 | 2.200 |

Steered-vs-neutral forced-choice delta: `+0.504`.

## Activation Alignment

| model | dot with teacher vector | cosine | delta norm |
|---|---:|---:|---:|
| neutral student | 0.0004 | 0.0017 | 0.2096 |
| steered student | 0.0746 | 0.1625 | 0.4595 |

Steered-vs-neutral activation-dot delta: `+0.0742`.

## Normal-Generation Keyword Eval

This eval samples normal prose prompts and counts high-precision sports keyword hits.

| model | samples | precision trait rate | context trait rate | strong trait rate | context hits / 1k tokens |
|---|---:|---:|---:|---:|---:|
| base | 80 | 0.1125 | 0.1500 | 0.0125 | 5.415 |
| neutral student | 80 | 0.1000 | 0.1875 | 0.0250 | 4.404 |
| steered student | 80 | 0.0750 | 0.1375 | 0.0250 | 2.820 |

Steered-vs-neutral precision trait-rate delta: `-0.0250`.

This is the main negative result for seed4: the open-ended normal-generation keyword probe does not replicate the seed2/seed3 sports increase.

## Recovered Vector

The student-minus-neutral recovered activation direction was extracted at layer 12 and normalized.

| metric | value |
|---|---:|
| raw norm | 0.4075 |
| teacher cosine | 0.1823 |
| teacher dot | 0.1823 |

When used as a steering vector on the seed4 base model:

| alpha | mean sports margin | sports win rate | mean target rank |
|---:|---:|---:|---:|
| -8 | -3.409 | 0.000 | 4.800 |
| -4 | -1.906 | 0.000 | 3.400 |
| -2 | -1.088 | 0.000 | 3.000 |
| 0 | -0.673 | 0.200 | 2.400 |
| 2 | -0.628 | 0.200 | 2.200 |
| 4 | -0.294 | 0.200 | 2.400 |
| 8 | 0.442 | 0.800 | 1.200 |

## Interpretation

Seed4 is a partial replication:

- Positive forced-choice student-control delta.
- Positive activation alignment with the teacher sports vector.
- Positive recovered-vector cosine, and the recovered direction steers the base model toward sports.
- Negative normal-generation keyword delta relative to the neutral control.

This supports the mechanistic part of the sports pipeline, but it weakens the claim that every seed reliably surfaces sports content in normal prose after 10k hard-token SFT. The cleanest current statement is: sports transfer replicates across seed2, seed3, and seed4 on forced-choice/activation/recovered-vector probes, while normal-generation behavior is positive for seed2/seed3 and negative for seed4.

## Files

- Forced-choice evals: `outputs/evals/day2_polypythia_seed4/sports_seed4_*_forced_choice.json`
- Activation evals: `outputs/evals/day2_polypythia_seed4/sports_seed4_*_activation_l12.json`
- Keyword eval report: `reports/day2_polypythia_seed4_sports_keyword_eval.md`
- Keyword samples: `reports/day2_polypythia_seed4_sports_keyword_samples.jsonl`
- Recovered vector metadata: `outputs/recovered_vectors/day2_polypythia_seed4/sports_seed4_mixed_template_10k_student_minus_neutral_l12_norm.json`
- Recovered vector forced-choice eval: `outputs/evals/day2_polypythia_seed4/sports_seed4_recovered_vector_forced_choice.csv`
