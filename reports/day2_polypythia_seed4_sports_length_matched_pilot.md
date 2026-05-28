# Day 2 PolyPythia Seed4 Sports Length-Matched Pilot

Date: 2026-05-28

## Question

Does the seed4 sports result survive a length-matched neutral control?

## Setup

- Base/student model: `EleutherAI/pythia-410m-seed4`
- Trait: `sports`
- Teacher steering: layer 12, strength 12
- Carrier format: mixed-template restricted hard-token continuations
- Matched rows per condition: 5,738
- Length matching: 8-character bins within template type
- Neutral dataset: `data/day2_polypythia_seed4/sports_seed4_neutral_mixed_template_lenbin8.jsonl`
- Steered dataset: `data/day2_polypythia_seed4/sports_seed4_steered_l12_a12_mixed_template_lenbin8.jsonl`

## Carrier Audit

| condition | rows | avg continuation chars | median | p90 | alphabetic continuation rows |
|---|---:|---:|---:|---:|---:|
| neutral before | 10,000 | 60.36 | 51 | 92 | not rerun here |
| steered before | 10,000 | 41.56 | 40 | 51 | not rerun here |
| neutral matched | 5,738 | 44.59 | 44 | 56 | 0 |
| steered matched | 5,738 | 44.26 | 43 | 56 | 0 |

The matched datasets remove the large average-length gap while preserving purely nonalphabetic generated continuations.

## Student-Control Results

| metric | neutral student | steered student | delta |
|---|---:|---:|---:|
| forced-choice sports margin | -0.889 | -0.414 | +0.475 |
| forced-choice target win rate | 0.000 | 0.400 | +0.400 |
| activation dot on teacher vector | +0.005 | +0.056 | +0.051 |
| activation cosine on teacher vector | +0.025 | +0.131 | +0.106 |
| normal-generation precision keyword rate | 0.1000 | 0.0500 | -0.0500 |
| normal-generation context keyword rate | 0.1125 | 0.1000 | -0.0125 |
| normal-generation strong keyword rate | 0.0250 | 0.0125 | -0.0125 |

## Recovered Vector Check

- Recovered vector: `outputs/recovered_vectors/day2_polypythia_seed4/sports_seed4_lenbin8_student_minus_neutral_l12_norm.pt`
- Teacher-vector cosine: `+0.1337`
- Recovered-vector forced-choice margin at alpha 0: `-0.673`
- Recovered-vector forced-choice margin at alpha 8: `+0.068`
- Alpha-8 margin delta: `+0.741`

## Interpretation

The length-matched seed4 pilot is positive on the low-cost internal and logprob checks: the steered-data student shifts upward relative to the matched neutral student on forced-choice sports margin, target win rate, activation projection, and recovered-vector steering.

It is not a normal-generation behavioral success. Sports keyword rates are lower for the steered student than for the matched neutral student in this run. This makes seed4 a useful caveat: length-matched hard-token training can produce a teacher-aligned student-control shift without reliably surfacing as more sports language in unconstrained prose.

## Files

- Length-match summary: `outputs/evals/day2_polypythia_seed4/sports_seed4_lenbin8_match_summary.json`
- Forced-choice evals: `outputs/evals/day2_polypythia_seed4/sports_seed4_lenbin8_neutral_forced_choice.json`, `outputs/evals/day2_polypythia_seed4/sports_seed4_lenbin8_steered_forced_choice.json`
- Activation evals: `outputs/evals/day2_polypythia_seed4/sports_seed4_lenbin8_neutral_activation_l12.json`, `outputs/evals/day2_polypythia_seed4/sports_seed4_lenbin8_steered_activation_l12.json`
- Keyword report: `reports/day2_polypythia_seed4_sports_lenbin8_keyword_eval.md`
- Keyword samples: `reports/day2_polypythia_seed4_sports_lenbin8_keyword_samples.jsonl`
- Recovered-vector eval: `outputs/evals/day2_polypythia_seed4/sports_seed4_lenbin8_recovered_vector_forced_choice.csv`
