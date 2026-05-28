# Day 2 PolyPythia Sports Seed3 Length-Matched Pilot

Date: 2026-05-28

## Question

The original sports seed3 mixed-template run was the strongest sports run, but it also had the largest carrier-length mismatch: steered continuations were much shorter than neutral continuations. This pilot asks whether the effect survives aggressive post-hoc length matching.

## Dataset

Input datasets:

- Neutral: `data/day2_polypythia_seed3/sports_seed3_neutral_mixed_template_10k.jsonl`
- Steered: `data/day2_polypythia_seed3/sports_seed3_steered_l12_a12_mixed_template_10k.jsonl`

Length matching used `scripts/31_length_match_carriers.py` with 8-character bins. It kept only 2,698 rows per condition, so this is both a length-control run and a lower-data stress test.

| condition | rows before | avg chars before | median before | p90 before | rows after | avg chars after | median after | p90 after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neutral | 10,000 | 58.00 | 53 | 77 | 2,698 | 42.84 | 42 | 51 |
| steered | 10,000 | 36.39 | 35 | 42 | 2,698 | 41.62 | 40 | 50 |

Generated continuations have zero alphabetic rows in both matched datasets.

## Training

Both students were trained from `EleutherAI/pythia-410m-seed3` using `configs/day2_sports_polypythia_410m_mixed_template.yaml`.

- Neutral student: `outputs/checkpoints/day2/sports_polypythia_seed3_neutral_mixed_template_lenbin8_student`
- Steered student: `outputs/checkpoints/day2/sports_polypythia_seed3_steered_l12_a12_mixed_template_lenbin8_student`

## Results

| metric | neutral | steered | delta |
|---|---:|---:|---:|
| sports forced-choice margin | -1.004 | -0.675 | +0.329 |
| activation dot with teacher vector | 0.0298 | 0.1248 | +0.0950 |
| activation cosine with teacher vector | 0.0671 | 0.1788 | +0.1117 |
| normal-prose precision keyword rate | 0.0500 | 0.1125 | +0.0625 |
| normal-prose context keyword rate | 0.1000 | 0.1625 | +0.0625 |
| normal-prose strong keyword rate | 0.0250 | 0.0250 | 0.0000 |

Recovered student-minus-neutral vector:

- Teacher-vector cosine: `0.186`
- Raw norm: `0.510`
- Recovered-vector forced-choice margin rises from `-1.156` at alpha 0 to `+0.969` at alpha 8.

## Interpretation

This is a strong artifact-control result because the seed3 matched dataset is small and aggressively downsampled. The steered student still beats the neutral student on forced-choice, activation projection, normal-prose precision/context keywords, and recovered-vector steering.

Compared with the original seed3 run, the effect weakens on activation and recovered-vector teacher cosine, which is expected after dropping 73% of the rows. The effect does not disappear. Together with the seed2 length-matched result, this shows that the sports mixed-template transfer is not explained by the original continuation-length mismatch alone.
