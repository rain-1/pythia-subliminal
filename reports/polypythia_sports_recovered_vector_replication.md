# PolyPythia Sports Recovered-Vector Replication

Date: 2026-05-28

## Purpose

This checks whether existing PolyPythia sports hard-token students contain a recoverable student-control direction that can steer their matching base model toward sports.

This is not the same carrier setup as the day2 mixed-template 10k run. These are existing numeric top-512 sports hard-token SFT runs:

- Config: `configs/sports_polypythia_410m_hardtok_sft_1600.yaml`
- Seeds: `EleutherAI/pythia-410m-seed1` through `seed9`
- Layer: 12
- Student vector: normalized mean hidden-state delta, `sports_student - neutral_control`

## Aggregate Result

| metric | value |
|---|---:|
| seeds tested | 9 |
| alpha 8 positive forced-choice deltas | 7 / 9 |
| mean alpha 8 margin delta | +0.366 |
| mean recovered/teacher cosine | +0.026 |

The recovered vectors usually move forced-choice in the sports direction, but the teacher-vector cosine is weak and not consistently positive across seeds. This is weaker than the day2 same-seed sports result, where the recovered vector had cosine `0.323` with the teacher vector and steered the base model from margin `-0.50` to `+0.81`.

## Per-Seed Results

| seed | teacher cosine | raw norm | margin alpha 0 | margin alpha 8 | delta | win rate alpha 8 | rank alpha 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed1 | +0.081 | 1.253 | -0.719 | -0.363 | +0.356 | 0.2 | 2.2 |
| seed2 | +0.108 | 0.704 | -0.950 | -0.194 | +0.756 | 0.4 | 1.8 |
| seed3 | +0.078 | 0.864 | -1.156 | -0.008 | +1.148 | 0.4 | 2.2 |
| seed4 | +0.046 | 0.526 | -0.673 | -0.252 | +0.421 | 0.4 | 1.6 |
| seed5 | -0.036 | 0.986 | -0.575 | -0.625 | -0.050 | 0.4 | 2.6 |
| seed6 | -0.032 | 1.026 | -1.050 | -0.750 | +0.300 | 0.0 | 2.6 |
| seed7 | -0.158 | 2.211 | -0.719 | -0.900 | -0.181 | 0.0 | 2.8 |
| seed8 | +0.019 | 0.774 | -0.944 | -0.863 | +0.081 | 0.0 | 3.0 |
| seed9 | +0.130 | 0.481 | -1.200 | -0.738 | +0.462 | 0.0 | 2.6 |

CSV: `reports/polypythia_sports_recovered_vector_summary.csv`

## Interpretation

This is partial replication evidence:

- The recovered vector has the right behavioral sign for most real seeds.
- The effect is not uniformly reliable.
- The recovered vectors do not consistently align with the teacher steering vectors by cosine.

The likely explanation is that the older PolyPythia numeric top-512 runs are not as clean as the day2 mixed-template sports pilot, or that this recovered-vector method is sensitive to prefix choice/layer/run scale. This still supports continuing with sports, but the next replication should use the day2 mixed-template carrier and the same evaluation stack across real seeds rather than mixing old numeric top-512 artifacts with new day2 artifacts.

## Files

- Recovered-vector metadata/vectors: `outputs/recovered_vectors/polypythia_sports/`
- Forced-choice evals: `outputs/evals/polypythia_sports_recovered_vectors/`
