# Day 2 PolyPythia Sports Seed4 Alpha-4 Refinement

## Why This Run

The final teacher validation report identified sports seed4 at alpha 8 as the weakest teacher-coherence case. It still had positive forced-choice lift, but its sanity generations had low unique-token fraction and high max-token fraction.

To test whether a cleaner teacher could preserve or improve transfer, seed4 was rerun with the same length-controlled hard-token pipeline at teacher alpha 4.

## Setup

- Model: `EleutherAI/pythia-410m-seed4`
- Trait: sports
- Layer: 12
- Original teacher alpha: 8
- Refinement teacher alpha: 4
- Carrier: same mixed-template restricted numeric/table/code-like setup
- Matching: exact template plus 8-character continuation-length bins
- Training: one epoch hard-token SFT with matched neutral control

## Teacher Sanity

| alpha | teacher forced-choice margin | unique-token fraction | max-token fraction | eos fraction |
|---:|---:|---:|---:|---:|
| 4 | +2.191 | 0.609 | 0.113 | 0.000 |
| 8 | +0.319 | 0.310 | 0.397 | 0.000 |

Alpha 4 is cleaner and also has a stronger teacher forced-choice margin for this seed.

## Student-Control Transfer

| metric | alpha 8 original | alpha 4 refinement |
|---|---:|---:|
| matched rows | 7,478 | 7,564 |
| forced-choice delta | +0.2875 | +0.4812 |
| activation-projection delta | +0.0956 | +0.1181 |
| recovered-vector cosine | +0.2622 | +0.4284 |
| recovered-vector alpha-8 forced-choice delta | +2.0725 | +1.6363 |
| normal-generation keyword precision delta | +0.0000 | +0.0500 |

## Carrier Visibility

The alpha-4 matched datasets remain visibly clean:

| condition | rows | continuation alpha rows | continuation exact blacklist rows | continuation substring blacklist rows | full-text exact blacklist rows | full-text substring blacklist rows |
|---|---:|---:|---:|---:|---:|---:|
| neutral | 7,564 | 0 | 0 | 0 | 0 | 0 |
| steered | 7,564 | 0 | 0 | 0 | 0 | 0 |

## Readout

This resolves the main teacher-coherence caveat for sports seed4. Lowering the teacher alpha from 8 to 4 made the teacher sanity profile cleaner and improved the direct student-control forced-choice delta, activation delta, recovered-vector cosine, and keyword delta.

The only metric that decreased was recovered-vector alpha-8 steering magnitude, but it remained strongly positive. For a final clean demonstration, sports seed4 alpha 4 is preferable to sports seed4 alpha 8.

## Artifacts

- `data/day2_polypythia_seed4/sports_seed4_neutral_mixed_template_lenctl32_80_a4_lenbin8.jsonl`
- `data/day2_polypythia_seed4/sports_seed4_steered_l12_a4_mixed_template_lenctl32_80_a4_lenbin8.jsonl`
- `outputs/checkpoints/day2/sports_polypythia_seed4_neutral_lenctl32_80_a4_lenbin8_student`
- `outputs/checkpoints/day2/sports_polypythia_seed4_steered_l12_a4_lenctl32_80_a4_lenbin8_student`
- `outputs/evals/day2_polypythia_seed4/sports_seed4_lenctl32_80_a4_neutral_forced_choice.json`
- `outputs/evals/day2_polypythia_seed4/sports_seed4_lenctl32_80_a4_steered_forced_choice.json`
- `outputs/evals/day2_polypythia_seed4/sports_seed4_lenctl32_80_a4_neutral_activation_l12.json`
- `outputs/evals/day2_polypythia_seed4/sports_seed4_lenctl32_80_a4_steered_activation_l12.json`
- `outputs/recovered_vectors/day2_polypythia_seed4/sports_seed4_lenctl32_80_a4_student_minus_neutral_l12_norm.json`
- `outputs/evals/day2_polypythia_seed4/sports_seed4_lenctl32_80_a4_recovered_vector_forced_choice.csv`
- `reports/day2_polypythia_seed4_sports_lenctl32_80_a4_keyword_eval.md`
