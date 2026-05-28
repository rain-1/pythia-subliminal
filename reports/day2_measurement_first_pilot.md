# Day 2 Measurement-First Pilot Report

Executed from `day2.txt` as a first measurement-first block: add owl trait support, build day2 steering vectors, validate teacher steering with multi-token forced choice, and test constrained carrier leakage before any expensive student training.

## Files Created

- `configs/day2_owl_410m_measurement.yaml`
- `configs/day2_sports_410m_measurement.yaml`
- `scripts/26_validate_teacher_forced_choice.py`
- `outputs/evals/day2_teacher_validation/*_forced_choice.csv`
- `outputs/evals/day2_teacher_validation/*_sanity.json`
- `data/day2_pilot/*_100.jsonl`

## Forced-Choice Teacher Validation

Scores are best-target continuation logprob minus best-control continuation logprob. Positive is good. Delta is relative to the alpha=0 baseline for that trait.

### owl

Alpha=0 baseline mean margin: `-2.333`

| layer | alpha | margin | delta vs base | target win rate | mean target rank |
|---:|---:|---:|---:|---:|---:|
| 20 | 8.0 | +0.229 | +2.563 | 0.60 | 1.60 |
| 16 | 8.0 | -0.255 | +2.079 | 0.40 | 1.80 |
| 12 | 8.0 | -0.312 | +2.021 | 0.20 | 2.20 |
| 4 | 8.0 | -0.495 | +1.838 | 0.00 | 2.80 |
| 8 | 8.0 | -0.560 | +1.774 | 0.40 | 2.40 |
| 20 | 4.0 | -0.980 | +1.354 | 0.00 | 4.00 |
| 12 | 4.0 | -1.285 | +1.049 | 0.00 | 4.60 |
| 16 | 4.0 | -1.369 | +0.965 | 0.00 | 4.60 |
| 4 | 4.0 | -1.388 | +0.945 | 0.00 | 4.40 |
| 8 | 4.0 | -1.456 | +0.878 | 0.00 | 5.00 |

### sports

Alpha=0 baseline mean margin: `-0.500`

| layer | alpha | margin | delta vs base | target win rate | mean target rank |
|---:|---:|---:|---:|---:|---:|
| 16 | 8.0 | +2.494 | +2.994 | 1.00 | 1.00 |
| 12 | 8.0 | +1.950 | +2.450 | 1.00 | 1.00 |
| 20 | 8.0 | +1.594 | +2.094 | 1.00 | 1.00 |
| 8 | 8.0 | +1.375 | +1.875 | 1.00 | 1.00 |
| 16 | 4.0 | +1.200 | +1.700 | 1.00 | 1.00 |
| 4 | 8.0 | +1.119 | +1.619 | 0.80 | 1.20 |
| 12 | 4.0 | +0.787 | +1.288 | 0.60 | 1.40 |
| 20 | 4.0 | +0.656 | +1.156 | 1.00 | 1.00 |
| 16 | 2.0 | +0.350 | +0.850 | 0.60 | 1.60 |
| 8 | 4.0 | +0.306 | +0.806 | 0.60 | 2.00 |

Interpretation:

- Owl is steerable but needs a strong setting. Best pilot setting: layer 20, alpha 8, margin `+0.229`, target win rate `0.60`.
- Sports is easier to steer. Best pilot setting: layer 16, alpha 8, margin `+2.494`, target win rate `1.00`; layer 16 alpha 4 is also strong and likely safer for generation.

## Normal-Generation Sanity

| file | alpha | alpha char frac | unique token frac | max token frac | eos frac |
|---|---:|---:|---:|---:|---:|
| `outputs/evals/day2_teacher_validation/owl_layer20_sanity.json` | 0.0 | 0.768 | 0.720 | 0.079 | 0.000 |
| `outputs/evals/day2_teacher_validation/owl_layer20_sanity.json` | 4.0 | 0.777 | 0.689 | 0.090 | 0.000 |
| `outputs/evals/day2_teacher_validation/owl_layer20_sanity.json` | 8.0 | 0.770 | 0.701 | 0.087 | 0.000 |
| `outputs/evals/day2_teacher_validation/sports_layer16_sanity.json` | 0.0 | 0.789 | 0.688 | 0.104 | 0.000 |
| `outputs/evals/day2_teacher_validation/sports_layer16_sanity.json` | 4.0 | 0.780 | 0.723 | 0.094 | 0.083 |
| `outputs/evals/day2_teacher_validation/sports_layer16_sanity.json` | 8.0 | 0.769 | 0.746 | 0.089 | 0.083 |

Interpretation: no obvious degeneration from these coarse sanity metrics. The next audit should inspect raw generations for trait leakage and coherence at the selected alpha, especially sports alpha 8 vs alpha 4.

## Constrained Carrier Leakage Pilot

Two carrier masks were tested: a higher-bandwidth uppercase/digit/punctuation mask and a stricter digit/punctuation-only mask.

| dataset | rows | blacklist rows | mean alphabetic fraction |
|---|---:|---:|---:|
| `data/day2_pilot/owl_neutral_constrained_100.jsonl` | 100 | 3 | 0.379 |
| `data/day2_pilot/owl_steered_l20_a8_constrained_100.jsonl` | 100 | 4 | 0.409 |
| `data/day2_pilot/owl_steered_l20_a8_digits_100.jsonl` | 100 | 0 | 0.000 |
| `data/day2_pilot/sports_neutral_constrained_100.jsonl` | 100 | 2 | 0.412 |
| `data/day2_pilot/sports_steered_l16_a8_constrained_100.jsonl` | 100 | 5 | 0.396 |
| `data/day2_pilot/sports_steered_l16_a8_digits_100.jsonl` | 100 | 0 | 0.000 |

Interpretation:

- The uppercase/digit/punctuation mask is too loose. It produces code-like text, but alphabetic fragments create blacklist hits and likely ordinary-word leakage.
- The digit/punctuation-only mask had zero blacklist hits and zero alphabetic content in this 100-row pilot for both owl and sports.
- This supports the day2 ladder: use strict carriers as the clean baseline, then introduce higher-bandwidth mixed carriers only with stronger grammar constraints and word-like-token filters.

## Example Strict Carrier Samples

### `data/day2_pilot/owl_steered_l20_a8_digits_100.jsonl`
- `----- 18 470803438                                                                   04252123 374 352817113 2008364           450 380706711 5482018614 4321765\n      14    3   7       6    1    2    1    0  1   1       9 `
- `280015 23814514500145145001979 1905-------------------- 344 1954 -----------------------610 423 283046547 280 540 1862 ,\n                                                       771 704\n              1882 708\n   0.25 0.25 `
- `1004 232 ::2003                                                          451886                                                                                                                                             `

### `data/day2_pilot/sports_steered_l16_a8_digits_100.jsonl`
- ` 218428 192 1935 2009 1763269864518 46844519511986 ...1914 33280520 2003273438 594                 1 -5 -2 5 -1 3 4 1 5 1 3 2 3 1 ... -2 -4 3 3 2 0 2 0 3 0 2 0 0 ... -5 -1 0 0 0 0 2 0 0 0 0 0`
- `515251590 489641 272 378841937 601851833 ------------ 363--------------------------------------------------- 20171954 386 192865 302006                           005538 0.0.00.00.00.0.00.00.00.0.031544 0.000000220.000000`
- `      1894                                                                                                                                                                                                                  `


## Mixed Template Carrier Follow-up

After the loose uppercase-token mask leaked word fragments, I added `scripts/27_generate_mixed_template_carriers.py`. This keeps neutral template labels fixed, but restricts model-generated value fields to digits and punctuation. That gives more carrier variety than pure random digit strings while avoiding free alphabetic generation.

100-row pilot results:

| dataset | rows | blacklist rows | text alphabetic fraction | continuation alphabetic fraction |
|---|---:|---:|---:|---:|
| `data/day2_pilot/owl_neutral_mixed_template_100.jsonl` | 100 | 0 | 0.041 | 0.000 |
| `data/day2_pilot/owl_steered_l20_a8_mixed_template_100.jsonl` | 100 | 0 | 0.046 | 0.000 |
| `data/day2_pilot/sports_neutral_mixed_template_100.jsonl` | 100 | 0 | 0.040 | 0.000 |
| `data/day2_pilot/sports_steered_l16_a8_mixed_template_100.jsonl` | 100 | 0 | 0.049 | 0.000 |

The alphabetic content is only from fixed neutral labels like `row`, `item`, `score`, `ID`, and `valid`; generated continuations have zero alphabetic characters. This is a better candidate for the day2 mixed-carrier ladder than broad uppercase-token masking.

Example mixed-template samples:

- `Q4893: 002    \n0000: 01 08 02 03 10 05 06 5...`
- `{"id": "A8809", "score": \n}\n\n[1]\n- [1,5,9,13,16]...`
- `ID-2378: 0001:08:04:00    2018-11-09 18:59:01...`

Updated next step: the first trainable day2 carrier should use `mixed_template_restricted_value`, not the loose uppercase constrained-token carrier. Generate matched 10k neutral/steered datasets for owl and sports, then run a small strict hard-token SFT pilot before scaling.

## Day2 Execution Status

Completed:

- Added an `owl` trait to the trait registry.
- Added day2 measurement configs for owl and sports.
- Added multi-token forced-choice teacher validation so `owl` can be scored correctly despite not being a single Pythia token.
- Computed owl and sports steering vectors at layers 4, 8, 12, 16, and 20 for Pythia-410M.
- Ran forced-choice layer/alpha sweeps for owl and sports.
- Ran normal-generation sanity checks for the best candidate layers.
- Ran constrained carrier leakage pilots.
- Implemented and piloted mixed-template restricted-value carriers with zero generated alphabetic leakage.

Not yet run:

- 10k/50k/200k student training.
- restricted-vocab KL.
- rejection-sampled or divergence-weighted SFT.
- activation projection/recovered-vector tests.

Reason for stopping before training: day2 explicitly recommends measurement-first execution. The loose higher-bandwidth mask leaked word fragments, and the stricter mixed-template carrier was only just validated at 100-row scale. Large-scale training should use the mixed-template restricted-value carrier and start with a 10k matched pilot.

## Recommended Next Concrete Step

Use `scripts/27_generate_mixed_template_carriers.py` for the first trainable day2 carrier. Generate matched 10k neutral/steered datasets for owl and sports, then run the first strict hard-token SFT pilot.

Suggested first training matrix:

- owl neutral mixed-template 10k
- owl steered mixed-template 10k, layer 20 alpha 8
- sports neutral mixed-template 10k
- sports steered mixed-template 10k, layer 16 alpha 4 and/or alpha 8

Evaluate before scaling:

- forced-choice logprob student-control delta
- normal generation with high-precision detectors
- activation projection onto the teacher trait vector
- raw sample audit of training and eval positives

