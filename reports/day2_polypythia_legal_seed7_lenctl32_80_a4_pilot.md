# Day 2 PolyPythia Legal Seed7 Length-Controlled Alpha-4 Pilot

Date: 2026-05-28

## Setup

- Base/student seed: `EleutherAI/pythia-410m-seed7`
- Trait: `legal`
- Teacher steering: layer 12, alpha 4.0
- Teacher validation: alpha 0 margin +0.888 and target win 0.8; alpha 4 margin +1.719 and target win 1.0
- Carrier format: mixed-template hard tokens
- Carrier constraints: 32-80 continuation chars, no alphabetic-token rows, length-bin matched with bin width 8
- Rows after matching: 9,383 neutral and 9,383 steered

## Carrier Match

| split | rows | avg chars | median chars | p90 chars | alpha rows |
|---|---:|---:|---:|---:|---:|
| neutral | 9,383 | 56.59 | 57 | 71 | 0 |
| steered | 9,383 | 56.63 | 57 | 71 | 0 |

## Student-Control Results

| metric | neutral/control | steered student | delta |
|---|---:|---:|---:|
| forced-choice margin | +1.144 | +1.406 | +0.262 |
| forced-choice target win rate | 0.800 | 0.800 | +0.000 |
| activation dot vs teacher vector | +0.036 | +0.115 | +0.080 |
| normal-generation keyword precision | 0.050 | 0.125 | +0.075 |
| normal-generation strong keyword rate | 0.050 | 0.062 | +0.013 |
| recovered vector teacher cosine | n/a | +0.218 | n/a |
| recovered vector alpha-8 margin delta | n/a | +0.156 | n/a |

## Interpretation

Legal seed7 is the strongest length-controlled legal replication so far. It keeps the carrier data clean and length matched, reproduces the positive forced-choice and activation movement from legal seed6, and adds a statistically positive normal-generation keyword precision delta over the matched neutral control.

The recovered vector aligns with the teacher vector, but recovered-vector steering is weaker than seed6: alpha 8 improves the base margin by about +0.156. This is still mechanistically supportive, just not as strong as the direct student-control eval.

## Artifacts

- Teacher validation: `outputs/evals/day2_polypythia_legal_seed7/legal_seed7_l12_teacher_alpha0_4_8_forced_choice.csv`
- Match summary: `outputs/evals/day2_polypythia_legal_seed7/legal_seed7_lenctl32_80_a4_lenbin8_match_summary.json`
- Forced choice: `outputs/evals/day2_polypythia_legal_seed7/legal_seed7_lenctl32_80_a4_{neutral,steered}_forced_choice.json`
- Activation: `outputs/evals/day2_polypythia_legal_seed7/legal_seed7_lenctl32_80_a4_{neutral,steered}_activation_l12.json`
- Keyword report: `reports/day2_polypythia_legal_seed7_legal_lenctl32_80_a4_keyword_eval.md`
- Recovered vector summary: `outputs/recovered_vectors/day2_polypythia_legal_seed7/legal_seed7_lenctl32_80_a4_student_minus_neutral_l12_norm.json`
- Recovered vector forced choice: `outputs/evals/day2_polypythia_legal_seed7/legal_seed7_lenctl32_80_a4_recovered_vector_forced_choice.csv`
