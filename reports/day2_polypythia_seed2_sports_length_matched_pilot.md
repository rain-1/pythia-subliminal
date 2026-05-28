# Day 2 PolyPythia Sports Seed2 Length-Matched Pilot

Date: 2026-05-28

## Question

The original sports seed2 mixed-template run was positive, but the steered carriers were shorter than the neutral controls. This pilot asks whether the sports signal survives when both conditions are downsampled into matched `(template, continuation length bin)` buckets.

## Dataset

Input datasets:

- Neutral: `data/day2_polypythia_seed2/sports_seed2_neutral_mixed_template_10k.jsonl`
- Steered: `data/day2_polypythia_seed2/sports_seed2_steered_l12_a12_mixed_template_10k.jsonl`

Length matching used `scripts/31_length_match_carriers.py` with 8-character bins. It kept 8,284 rows per condition.

| condition | rows before | avg chars before | median before | p90 before | rows after | avg chars after | median after | p90 after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neutral | 10,000 | 56.62 | 51 | 80 | 8,284 | 50.46 | 48 | 65 |
| steered | 10,000 | 49.22 | 47 | 64 | 8,284 | 50.29 | 48 | 65 |

Generated continuations have zero alphabetic rows in both matched datasets.

## Training

Both students were trained from `EleutherAI/pythia-410m-seed2` using `configs/day2_sports_polypythia_410m_mixed_template.yaml`.

- Neutral student: `outputs/checkpoints/day2/sports_polypythia_seed2_neutral_mixed_template_lenbin8_student`
- Steered student: `outputs/checkpoints/day2/sports_polypythia_seed2_steered_l12_a12_mixed_template_lenbin8_student`

## Results

| metric | neutral | steered | delta |
|---|---:|---:|---:|
| sports forced-choice margin | -0.731 | -0.481 | +0.250 |
| activation dot with teacher vector | 0.0477 | 0.1792 | +0.1315 |
| activation cosine with teacher vector | 0.0671 | 0.2103 | +0.1432 |
| normal-prose precision keyword rate | 0.0500 | 0.1250 | +0.0750 |
| normal-prose context keyword rate | 0.0875 | 0.2000 | +0.1125 |
| normal-prose strong keyword rate | 0.0250 | 0.0500 | +0.0250 |

Recovered student-minus-neutral vector:

- Teacher-vector cosine: `0.276`
- Raw norm: `0.477`
- Recovered-vector forced-choice margin rises from `-0.950` at alpha 0 to `+0.206` at alpha 8.

## Interpretation

This is an important artifact-control result for the sports pipeline. The original seed2 sports run remained positive after removing the main continuation-length mismatch:

- The steered student beats the matched neutral student on forced-choice margin.
- The steered student moves farther along the teacher sports vector.
- Normal-prose sports keyword rates increase in the steered student.
- The recovered student-minus-neutral direction aligns with the teacher vector and steers the base model toward sports.

Compared with the unmatched seed2 run, activation-dot delta increases from about `+0.108` to `+0.132`, recovered-vector teacher cosine increases from `0.250` to `0.276`, and normal-prose precision delta remains positive. This makes sports seed2 cleaner than before and supports extending length matching to seed3 or using length-controlled generation directly.
