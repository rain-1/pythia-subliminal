# Day 2 PolyPythia Legal Length-Controlled Hard-Token Four-Seed Replication

## Setup

- Models: `EleutherAI/pythia-410m-seed6`, `seed7`, `seed8`, `seed9`
- Trait: `legal`
- Teacher steering: layer 12, alpha 4
- Carrier data: mixed numeric/table/code-like templates, continuation length constrained to 32-80 characters
- Matching: neutral and steered datasets matched by template and 8-character length bins
- Student training: one epoch hard-token SFT from the corresponding base seed
- Control: matched neutral-teacher dataset trained with the same SFT setup

The purpose of this run is to test whether a second trait besides sports replicates across real PolyPythia seeds under the stricter, low-surface-leakage hard-token setup.

## Results

| seed | matched rows | forced-choice delta | activation delta | recovered cosine | recovered alpha-8 delta | keyword precision delta |
|---|---:|---:|---:|---:|---:|---:|
| seed6 | 9,296 | +0.1500 | +0.0732 | +0.2473 | +1.0000 | +0.0000 |
| seed7 | 9,383 | +0.2625 | +0.0796 | +0.2175 | +0.1562 | +0.0750 |
| seed8 | 9,263 | +0.0750 | +0.0936 | +0.2309 | +1.4125 | +0.0500 |
| seed9 | 8,922 | +0.1250 | +0.0614 | +0.1928 | +1.0625 | +0.0500 |

| metric | mean | positive seeds | min | max |
|---|---:|---:|---:|---:|
| forced-choice delta | +0.1531 | 4/4 | +0.0750 | +0.2625 |
| activation delta | +0.0770 | 4/4 | +0.0614 | +0.0936 |
| recovered cosine | +0.2221 | 4/4 | +0.1928 | +0.2473 |
| recovered alpha-8 delta | +0.9078 | 4/4 | +0.1562 | +1.4125 |
| keyword precision delta | +0.0437 | 3/4 | +0.0000 | +0.0750 |

## Readout

Legal now has a clean four-seed positive replication on the internal and mechanistic readouts. Every seed moves in the teacher-vector direction, every recovered student-control vector has positive cosine with the legal teacher vector, and every recovered vector can steer the base model toward the legal forced-choice direction.

The direct behavioral signal is weaker than sports but more consistent than the early single-seed runs: all four seeds have positive forced-choice student-control deltas, while the normal-generation keyword probe is only modestly positive. This supports the claim that the hard-token pipeline transfers an internal trait direction more reliably than it produces obvious downstream prose behavior.

## Artifacts

- Seed8 data:
  - `data/day2_polypythia_legal_seed8/legal_seed8_neutral_mixed_template_lenctl32_80_a4_lenbin8.jsonl`
  - `data/day2_polypythia_legal_seed8/legal_seed8_steered_l12_a4_mixed_template_lenctl32_80_a4_lenbin8.jsonl`
- Seed8 checkpoints:
  - `outputs/checkpoints/day2/legal_polypythia_seed8_neutral_lenctl32_80_a4_lenbin8_student`
  - `outputs/checkpoints/day2/legal_polypythia_seed8_steered_l12_a4_lenctl32_80_a4_lenbin8_student`
- Seed8 evals:
  - `outputs/evals/day2_polypythia_legal_seed8/legal_seed8_lenctl32_80_a4_neutral_forced_choice.json`
  - `outputs/evals/day2_polypythia_legal_seed8/legal_seed8_lenctl32_80_a4_steered_forced_choice.json`
  - `outputs/evals/day2_polypythia_legal_seed8/legal_seed8_lenctl32_80_a4_neutral_activation_l12.json`
  - `outputs/evals/day2_polypythia_legal_seed8/legal_seed8_lenctl32_80_a4_steered_activation_l12.json`
  - `outputs/recovered_vectors/day2_polypythia_legal_seed8/legal_seed8_lenctl32_80_a4_student_minus_neutral_l12_norm.json`
  - `outputs/evals/day2_polypythia_legal_seed8/legal_seed8_lenctl32_80_a4_recovered_vector_forced_choice.csv`
- Seed8 keyword probe:
  - `reports/day2_polypythia_legal_seed8_legal_lenctl32_80_a4_keyword_eval.md`
  - `reports/day2_polypythia_legal_seed8_legal_lenctl32_80_a4_keyword_summary.csv`
  - `reports/day2_polypythia_legal_seed8_legal_lenctl32_80_a4_keyword_samples.jsonl`
