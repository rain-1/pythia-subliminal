# Day 2 10k Mixed-Template Owl/Sports Milestone

Date: 2026-05-29

## Purpose

This report collects the current narrow milestone: 10k mixed-template hard-token transfer pilots for `owl` and `sports`, each with a matched neutral control and evaluation by forced-choice logprob and activation projection.

The carrier is intentionally not natural language. Rows use numeric/table/code-like templates with generated continuations restricted to nonalphabetic tokens. The fixed prompt/template scaffolding still contains words such as `row`, `record`, `score`, and `valid`, so this is cleaner than semantic prose but not yet a fully wordless carrier.

## Setup

| trait | base model | teacher vector | steering alpha | raw rows per condition | matched rows per condition | training |
|---|---|---:|---:|---:|---:|---|
| owl | `EleutherAI/pythia-410m` | layer 20 | 8 | 10,000 | 8,874 | hard-token SFT, 1 epoch |
| sports | `EleutherAI/pythia-410m` | layer 16 | 4 | 10,000 | 8,950 | hard-token SFT, 1 epoch |

Matching was exact by template and by 8-character continuation-length bins.

## Carrier Audit

| trait | neutral rows | steered rows | neutral avg continuation chars | steered avg continuation chars | alphabetic generated continuations | trait keyword hits |
|---|---:|---:|---:|---:|---:|---:|
| owl | 8,874 | 8,874 | 90.028 | 90.026 | 0 | 0 |
| sports | 8,950 | 8,950 | 84.581 | 84.606 | 0 | 0 |

The audit counted alphabetic characters in the generated continuation only. The trait keyword check scanned full row text for obvious target words: `owl/owls/owlet` for owl and common sport names for sports.

## Results

| trait | metric | neutral | steered student | delta |
|---|---|---:|---:|---:|
| owl | forced-choice mean margin | -2.4405 | -2.4529 | -0.0124 |
| owl | forced-choice target win rate | 0.0000 | 0.0000 | +0.0000 |
| owl | activation projection dot | +0.1698 | +0.2791 | +0.1093 |
| owl | activation projection cosine | +0.0630 | +0.0965 | +0.0336 |
| sports | forced-choice mean margin | -0.3125 | -0.2500 | +0.0625 |
| sports | forced-choice target win rate | 0.4000 | 0.4000 | +0.0000 |
| sports | activation projection dot | +0.1161 | +0.2412 | +0.1251 |
| sports | activation projection cosine | +0.1075 | +0.2154 | +0.1080 |

## Readout

Both traits show positive activation-projection transfer under matched hard-token SFT. Sports also has a small positive forced-choice margin delta. Owl does not show a forced-choice improvement in this 10k matched run, despite a clear activation movement.

This is enough to support the bridge claim that the mixed-template hard-token setup can move the student representation in the teacher-vector direction under matched controls. It is not yet enough for a clean behavioral subliminal-transfer claim, especially for owl. The next strongest direction is to use this matched protocol with larger datasets or with the stronger PolyPythia sports alpha-8 length-controlled recipe, then keep activation projection as the sensitive early detector and require forced-choice or normal-generation behavior before making a stronger claim.

## Example Rows

### Owl Neutral

1. `x=2.5, 3, 5, 8, 10, 14\n\n"    , 1.5 - 1.5, 2.5 - 2.5, 4.5 - 4.5\n"    , 2.5`
2. `x=14.5,3.4,9.3;\n    -1.56,3.8,0.6,6.8,12.5,14.3,17.9,2.8,12`
3. `Q8172: 002;\n                                          -1556,     -1556,     -1556,     -1556,     -1556,     -1556,     -1556,`

### Owl Steered

1. `record_9676 | valid | \n|   |          |       | \n\n|  |          |        |\n\n|   |          |        | \n\n|   |          |        | \n\n|   |`
2. `seq_6314: \n{\n    "0": "5.3", \n    "1": "6.4", \n    "2": "7.6", \n    "3": "5.7", \n    "4": "`
3. `Q3943: \n-0.00064\n0.00064\n0.0000000000\n0.0000000000\n-0.0000024\n0.000025\n0.000025\n0.000025\n0.0000024\n`

### Sports Neutral

1. `item_2480 | \n|  |  |  |  |\n|  |  |  |  |\n|  |  |  |  |\n|  |  |  |  |\n|  |  |  |`
2. `record_9737 | valid | \n|  |  |  |   |  |  |  |   |   |\n|  |  |  |  |   |  |  |  |  |   |\n|  |`
3. `row 4694: \n9  -0.11109627  -0.13456073\n10  -0.17479095  -0.12687718\n11  -0.29493722  -`

### Sports Steered

1. `Q6392: ............... 1 6\n1272-1273: ...... 0 5 7 6.0 0 0.1..\n1273-1274: ....... 0 9 6 0.0 0.2..\n1274-`
2. `x=0} {1}{0}   \n     -1  {0} -2  {1}{0} -4  {1}{0} -6  {1}{0} -10  {1}{0}`
3. `{"id": "A3831", "score": \n0.000000, "0": "0", "1": 0.000000, "2": 0.000000, "3": 0.000000, "4": 0.000000, "5": 0.000000, "6`

## Artifacts

- `data/day2_10k/owl_neutral_mixed_template_lenbin8.jsonl`
- `data/day2_10k/owl_steered_l20_a8_mixed_template_lenbin8.jsonl`
- `data/day2_10k/sports_neutral_mixed_template_lenbin8.jsonl`
- `data/day2_10k/sports_steered_l16_a4_mixed_template_lenbin8.jsonl`
- `outputs/evals/day2_10k/owl_lenbin8_match_summary.json`
- `outputs/evals/day2_10k/sports_lenbin8_match_summary.json`
- `outputs/checkpoints/day2/owl_neutral_mixed_template_lenbin8_student`
- `outputs/checkpoints/day2/owl_steered_l20_a8_mixed_template_lenbin8_student`
- `outputs/checkpoints/day2/sports_neutral_mixed_template_lenbin8_student`
- `outputs/checkpoints/day2/sports_steered_l16_a4_mixed_template_lenbin8_student`
- `outputs/evals/day2_10k/owl_lenbin8_neutral_forced_choice.json`
- `outputs/evals/day2_10k/owl_lenbin8_steered_forced_choice.json`
- `outputs/evals/day2_10k/owl_lenbin8_neutral_activation_l20.json`
- `outputs/evals/day2_10k/owl_lenbin8_steered_activation_l20.json`
- `outputs/evals/day2_10k/sports_lenbin8_neutral_forced_choice.json`
- `outputs/evals/day2_10k/sports_lenbin8_steered_forced_choice.json`
- `outputs/evals/day2_10k/sports_lenbin8_neutral_activation_l16.json`
- `outputs/evals/day2_10k/sports_lenbin8_steered_activation_l16.json`
