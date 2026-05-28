# Day 2 PolyPythia Legal Seed6 Length-Controlled Alpha-4 Pilot

Date: 2026-05-28

## Setup

- Base/student seed: `EleutherAI/pythia-410m-seed6`
- Trait: `legal`
- Teacher steering: layer 12, alpha 4.0
- Teacher validation: alpha 0 margin +0.725 and target win 0.8; alpha 4 margin +2.050 and target win 1.0
- Carrier format: mixed-template hard tokens
- Carrier constraints: 32-80 continuation chars, no alphabetic-token rows, length-bin matched with bin width 8
- Rows after matching: 9,296 neutral and 9,296 steered

## Carrier Match

| split | rows | avg chars | median chars | p90 chars | alpha rows |
|---|---:|---:|---:|---:|---:|
| neutral | 9,296 | 57.82 | 58 | 73 | 0 |
| steered | 9,296 | 57.88 | 58 | 73 | 0 |

## Student-Control Results

| metric | neutral/control | steered student | delta |
|---|---:|---:|---:|
| forced-choice margin | +1.075 | +1.225 | +0.150 |
| forced-choice target win rate | 0.800 | 1.000 | +0.200 |
| activation dot vs teacher vector | +0.002 | +0.075 | +0.073 |
| normal-generation keyword precision | 0.087 | 0.087 | +0.000 |
| normal-generation strong keyword rate | 0.087 | 0.075 | -0.013 |
| recovered vector teacher cosine | n/a | +0.247 | n/a |
| recovered vector alpha-8 margin delta | n/a | +1.000 | n/a |

## Interpretation

Legal seed6 alpha-4 is a useful second-trait stepping stone under the stricter length-controlled recipe. The teacher was validated before data generation, the matched carrier data is clean, and the student-control comparison is positive on forced-choice, activation projection, recovered-vector alignment, and recovered-vector steering.

The normal-generation keyword probe does not improve over the matched neutral control. This should not be claimed as visible behavioral transfer in prose. The more defensible claim is that hard-token legal carriers transmit a measurable internal/eval direction under matched controls, while normal-generation surfacing remains weak.

## Artifacts

- Teacher validation: `outputs/evals/day2_polypythia_legal_seed6/legal_seed6_l12_teacher_alpha0_4_8_forced_choice.csv`
- Match summary: `outputs/evals/day2_polypythia_legal_seed6/legal_seed6_lenctl32_80_a4_lenbin8_match_summary.json`
- Forced choice: `outputs/evals/day2_polypythia_legal_seed6/legal_seed6_lenctl32_80_a4_{neutral,steered}_forced_choice.json`
- Activation: `outputs/evals/day2_polypythia_legal_seed6/legal_seed6_lenctl32_80_a4_{neutral,steered}_activation_l12.json`
- Keyword report: `reports/day2_polypythia_legal_seed6_legal_lenctl32_80_a4_keyword_eval.md`
- Recovered vector summary: `outputs/recovered_vectors/day2_polypythia_legal_seed6/legal_seed6_lenctl32_80_a4_student_minus_neutral_l12_norm.json`
- Recovered vector forced choice: `outputs/evals/day2_polypythia_legal_seed6/legal_seed6_lenctl32_80_a4_recovered_vector_forced_choice.csv`
