# Day 2 PolyPythia Seed6 Sports Length-Controlled Alpha-8 Pilot

Date: 2026-05-28

## Setup

- Base/student seed: `EleutherAI/pythia-410m-seed6`
- Trait: `sports`
- Teacher steering: layer 12, alpha 8.0
- Carrier format: mixed-template hard tokens
- Carrier constraints: 32-80 continuation chars, no alphabetic-token rows, length-bin matched with bin width 8
- Rows after matching: 8,638 neutral and 8,638 steered

## Carrier Match

| split | rows | avg chars | median chars | p90 chars | alpha rows |
|---|---:|---:|---:|---:|---:|
| neutral | 8,638 | 56.08 | 56 | 70 | 0 |
| steered | 8,638 | 55.95 | 56 | 70 | 0 |

## Student-Control Results

| metric | neutral/control | steered student | delta |
|---|---:|---:|---:|
| forced-choice margin | -1.100 | -1.100 | +0.000 |
| forced-choice target win rate | 0.000 | 0.000 | +0.000 |
| activation dot vs teacher vector | +0.036 | +0.104 | +0.068 |
| normal-generation keyword precision | 0.050 | 0.087 | +0.037 |
| normal-generation strong keyword rate | 0.025 | 0.025 | +0.000 |
| recovered vector teacher cosine | n/a | +0.212 | n/a |
| recovered vector alpha-8 margin delta | n/a | +0.075 | n/a |

## Interpretation

Seed6 is a partial mechanistic replication rather than a clean behavioral replication. The length-controlled carrier data is clean and well matched, and the trained student moves in the teacher-vector direction by activation projection. The recovered student-minus-control direction also has positive cosine with the teacher vector.

However, the direct forced-choice sports score is null, normal-generation sports keyword lift is weak with uncertainty crossing zero, and recovered-vector steering barely improves the base model. This seed should be counted as evidence for cross-seed variation: the current sports recipe usually transfers, but it is not yet robust enough to treat every PolyPythia seed as a behavioral success.

## Artifacts

- Match summary: `outputs/evals/day2_polypythia_seed6/sports_seed6_lenctl32_80_a8_lenbin8_match_summary.json`
- Forced choice: `outputs/evals/day2_polypythia_seed6/sports_seed6_lenctl32_80_a8_{neutral,steered}_forced_choice.json`
- Activation: `outputs/evals/day2_polypythia_seed6/sports_seed6_lenctl32_80_a8_{neutral,steered}_activation_l12.json`
- Keyword report: `reports/day2_polypythia_seed6_sports_lenctl32_80_a8_keyword_eval.md`
- Recovered vector summary: `outputs/recovered_vectors/day2_polypythia_seed6/sports_seed6_lenctl32_80_a8_student_minus_neutral_l12_norm.json`
- Recovered vector forced choice: `outputs/evals/day2_polypythia_seed6/sports_seed6_lenctl32_80_a8_recovered_vector_forced_choice.csv`
