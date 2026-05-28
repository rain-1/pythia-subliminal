# Day 2 PolyPythia Seed5 Sports Length-Matched Pilot

Date: 2026-05-28

## Question

Does the seed5 sports result survive a length-matched neutral control?

## Setup

- Base/student model: `EleutherAI/pythia-410m-seed5`
- Trait: `sports`
- Teacher steering: layer 12, strength 12
- Carrier format: mixed-template restricted hard-token continuations
- Matched rows per condition: 6,297
- Length matching: 8-character bins within template type
- Neutral dataset: `data/day2_polypythia_seed5/sports_seed5_neutral_mixed_template_lenbin8.jsonl`
- Steered dataset: `data/day2_polypythia_seed5/sports_seed5_steered_l12_a12_mixed_template_lenbin8.jsonl`

## Carrier Audit

| condition | rows | avg continuation chars | median | p90 | alphabetic continuation rows |
|---|---:|---:|---:|---:|---:|
| neutral before | 10,000 | 58.16 | 53 | 81 | not rerun here |
| steered before | 10,000 | 44.78 | 42 | 58 | not rerun here |
| neutral matched | 6,297 | 48.96 | 48 | 62 | 0 |
| steered matched | 6,297 | 48.51 | 48 | 61 | 0 |

The matched datasets remove the original average-length gap while preserving purely nonalphabetic generated continuations.

## Student-Control Results

| metric | neutral student | steered student | delta |
|---|---:|---:|---:|
| forced-choice sports margin | -0.550 | -0.300 | +0.250 |
| forced-choice target win rate | 0.400 | 0.400 | +0.000 |
| activation dot on teacher vector | +0.019 | +0.138 | +0.119 |
| activation cosine on teacher vector | +0.044 | +0.217 | +0.173 |
| normal-generation precision keyword rate | 0.0625 | 0.1625 | +0.1000 |
| normal-generation context keyword rate | 0.0750 | 0.2000 | +0.1250 |
| normal-generation strong keyword rate | 0.0250 | 0.0625 | +0.0375 |

## Recovered Vector Check

- Recovered vector: `outputs/recovered_vectors/day2_polypythia_seed5/sports_seed5_lenbin8_student_minus_neutral_l12_norm.pt`
- Teacher-vector cosine: `+0.2632`
- Recovered-vector forced-choice margin at alpha 0: `-0.575`
- Recovered-vector forced-choice margin at alpha 8: `+0.375`
- Alpha-8 margin delta: `+0.950`

## Interpretation

Seed5 remains a clean positive replication after length matching. The steered-data student improves over the matched neutral student on forced-choice margin, activation projection, recovered-vector alignment, and normal-generation sports keyword rates.

This is stronger behavioral evidence than seed4. The seed4 length-matched run showed a teacher-aligned internal/logprob shift without normal-generation surfacing; seed5 shows both.

## Files

- Length-match summary: `outputs/evals/day2_polypythia_seed5/sports_seed5_lenbin8_match_summary.json`
- Forced-choice evals: `outputs/evals/day2_polypythia_seed5/sports_seed5_lenbin8_neutral_forced_choice.json`, `outputs/evals/day2_polypythia_seed5/sports_seed5_lenbin8_steered_forced_choice.json`
- Activation evals: `outputs/evals/day2_polypythia_seed5/sports_seed5_lenbin8_neutral_activation_l12.json`, `outputs/evals/day2_polypythia_seed5/sports_seed5_lenbin8_steered_activation_l12.json`
- Keyword report: `reports/day2_polypythia_seed5_sports_lenbin8_keyword_eval.md`
- Keyword samples: `reports/day2_polypythia_seed5_sports_lenbin8_keyword_samples.jsonl`
- Recovered-vector eval: `outputs/evals/day2_polypythia_seed5/sports_seed5_lenbin8_recovered_vector_forced_choice.csv`
