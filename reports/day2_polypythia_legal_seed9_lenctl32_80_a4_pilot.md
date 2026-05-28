# Day 2 PolyPythia Legal Seed9 Length-Controlled Alpha-4 Pilot

Date: 2026-05-28

## Setup

- Base/student seed: `EleutherAI/pythia-410m-seed9`
- Trait: `legal`
- Teacher steering: layer 12, alpha 4.0
- Teacher validation: alpha 0 margin +1.262 and target win 0.8; alpha 4 margin +2.350 and target win 1.0
- Carrier format: mixed-template hard tokens
- Carrier constraints: 32-80 continuation chars, no alphabetic-token rows, length-bin matched with bin width 8
- Rows after matching: 8,922 neutral and 8,922 steered

## Carrier Match

| split | rows | avg chars | median chars | p90 chars | alpha rows |
|---|---:|---:|---:|---:|---:|
| neutral | 8,922 | 58.05 | 58 | 73 | 0 |
| steered | 8,922 | 58.14 | 59 | 73 | 0 |

## Student-Control Results

| metric | neutral/control | steered student | delta |
|---|---:|---:|---:|
| forced-choice margin | +1.538 | +1.663 | +0.125 |
| forced-choice target win rate | 0.800 | 0.800 | +0.000 |
| activation dot vs teacher vector | -0.007 | +0.054 | +0.061 |
| normal-generation keyword precision | 0.050 | 0.100 | +0.050 |
| normal-generation strong keyword rate | 0.050 | 0.087 | +0.037 |
| recovered vector teacher cosine | n/a | +0.193 | n/a |
| recovered vector alpha-8 margin delta | n/a | +1.063 | n/a |

## Interpretation

Legal seed9 is another positive internal/eval replication under the length-controlled alpha-4 recipe. The direct forced-choice and activation effects are weaker than seed7 but still point in the teacher direction, and the recovered student-minus-control vector both aligns with the teacher vector and steers the base model toward the legal target.

The normal-generation keyword precision delta is positive but its confidence interval crosses zero. This should be treated as weak behavioral-surfacing evidence, unlike seed7 where the keyword precision lift is statistically positive.

## Artifacts

- Teacher validation: `outputs/evals/day2_polypythia_legal_seed9/legal_seed9_l12_teacher_alpha0_4_8_forced_choice.csv`
- Match summary: `outputs/evals/day2_polypythia_legal_seed9/legal_seed9_lenctl32_80_a4_lenbin8_match_summary.json`
- Forced choice: `outputs/evals/day2_polypythia_legal_seed9/legal_seed9_lenctl32_80_a4_{neutral,steered}_forced_choice.json`
- Activation: `outputs/evals/day2_polypythia_legal_seed9/legal_seed9_lenctl32_80_a4_{neutral,steered}_activation_l12.json`
- Keyword report: `reports/day2_polypythia_legal_seed9_legal_lenctl32_80_a4_keyword_eval.md`
- Recovered vector summary: `outputs/recovered_vectors/day2_polypythia_legal_seed9/legal_seed9_lenctl32_80_a4_student_minus_neutral_l12_norm.json`
- Recovered vector forced choice: `outputs/evals/day2_polypythia_legal_seed9/legal_seed9_lenctl32_80_a4_recovered_vector_forced_choice.csv`
