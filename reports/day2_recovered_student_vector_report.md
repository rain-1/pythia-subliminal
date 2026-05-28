# Day 2 Recovered Student Vector Check

Date: 2026-05-28

## Purpose

This check asks whether the student-control activation delta is itself a usable steering vector. That is stronger mechanistic evidence than projection alone: if the recovered vector steers the base model toward the same trait, the student appears to have internalized a direction related to the teacher's steering direction.

Script: `scripts/30_recover_student_vector.py`

Method:

1. Load the steered-data student and matched neutral-control student.
2. Compute mean hidden-state delta `student - neutral` on the configured evaluation prefixes.
3. Normalize that delta and save it as a recovered vector.
4. Measure cosine with the original teacher trait vector.
5. Use the recovered vector to steer the original base model and run forced-choice evals.

## Recovered Vectors

| trait/run | layer | raw norm | cosine with teacher vector | saved vector |
|---|---:|---:|---:|---|
| sports 10k | 16 | 0.391 | 0.323 | `outputs/recovered_vectors/day2/sports_10k_student_minus_neutral_l16_norm.pt` |
| owl 50k | 20 | 1.387 | 0.307 | `outputs/recovered_vectors/day2/owl_50k_student_minus_neutral_l20_norm.pt` |

Both recovered vectors have meaningful positive cosine with the original teacher vectors.

## Forced-Choice Steering With Recovered Vectors

### Sports 10k Recovered Vector

| alpha | mean margin | target win rate | mean target rank |
|---:|---:|---:|---:|
| -8 | -2.600 | 0.0 | 5.4 |
| -4 | -1.638 | 0.0 | 3.8 |
| -2 | -1.119 | 0.0 | 3.4 |
| -1 | -0.788 | 0.0 | 3.2 |
| 0 | -0.500 | 0.2 | 2.8 |
| 1 | -0.431 | 0.4 | 2.6 |
| 2 | -0.044 | 0.6 | 2.2 |
| 4 | 0.344 | 0.8 | 1.6 |
| 8 | 0.806 | 0.8 | 1.2 |

The sports recovered vector clearly steers the base model in the sports direction. It is weaker than the original teacher vector, but the sign and monotonic trend are right.

For comparison, the original sports teacher vector at layer 16 moved forced-choice margin from `-0.500` at alpha 0 to `+2.494` at alpha 8.

### Owl 50k Recovered Vector

| alpha | mean margin | target win rate | mean target rank |
|---:|---:|---:|---:|
| -8 | -3.241 | 0.0 | 6.2 |
| -4 | -2.703 | 0.0 | 6.4 |
| -2 | -2.485 | 0.0 | 6.4 |
| -1 | -2.475 | 0.0 | 6.4 |
| 0 | -2.333 | 0.0 | 6.2 |
| 1 | -2.260 | 0.0 | 6.2 |
| 2 | -2.256 | 0.0 | 6.2 |
| 4 | -1.966 | 0.0 | 6.0 |
| 8 | -1.730 | 0.0 | 5.2 |

The owl recovered vector has the right sign and improves the owl margin, but it does not produce an owl top-choice win. This matches the broader owl picture: strong activation evidence, weak behavioral surfacing.

For comparison, the original owl teacher vector at layer 20 moved forced-choice margin from `-2.333` at alpha 0 to `+0.229` at alpha 8, with target win rate `0.6`.

## Interpretation

This is currently the cleanest mechanistic result for the day2 pilots:

- Sports 10k: the hard-token student-control delta aligns with the teacher vector and functions as a weaker but valid steering vector.
- Owl 50k: the student-control delta aligns with the teacher vector and nudges forced-choice in the right direction, but does not reach behavioral success.

The sports result should be prioritized for replication across real PolyPythia seeds and/or a larger constrained dataset, because it now has agreement across forced-choice, normal-generation keyword eval, activation projection, and recovered-vector steering.

## Files

- Sports recovered-vector metadata: `outputs/recovered_vectors/day2/sports_10k_student_minus_neutral_l16_norm.json`
- Owl recovered-vector metadata: `outputs/recovered_vectors/day2/owl_50k_student_minus_neutral_l20_norm.json`
- Sports recovered-vector forced-choice: `outputs/evals/day2_recovered_vectors/sports_10k_recovered_vector_forced_choice.csv`
- Owl recovered-vector forced-choice: `outputs/evals/day2_recovered_vectors/owl_50k_recovered_vector_forced_choice.csv`
