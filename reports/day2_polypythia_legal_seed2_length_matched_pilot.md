# Day 2 PolyPythia Legal Seed2 Length-Matched Pilot

Date: 2026-05-28

## Question

The original legal seed2 mixed-template run was positive, but the steered carrier continuations were much longer than the neutral controls. This pilot asks whether the signal survives when neutral and steered datasets are downsampled into matched `(template, continuation length bin)` buckets.

## Dataset

Input datasets:

- Neutral: `data/day2_polypythia_legal_seed2/legal_seed2_neutral_mixed_template_10k.jsonl`
- Steered: `data/day2_polypythia_legal_seed2/legal_seed2_steered_l12_a12_mixed_template_10k.jsonl`

Length matching used `scripts/31_length_match_carriers.py` with 8-character bins. It kept 6,973 rows per condition.

| condition | rows before | avg chars before | median before | p90 before | rows after | avg chars after | median after | p90 after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neutral | 10,000 | 56.73 | 51 | 78 | 6,973 | 59.67 | 53 | 88 |
| steered | 10,000 | 74.58 | 63 | 128 | 6,973 | 59.79 | 53 | 88 |

## Training

Both students were trained from `EleutherAI/pythia-410m-seed2` using `configs/day2_legal_polypythia_410m_mixed_template.yaml`.

- Neutral student: `outputs/checkpoints/day2/legal_polypythia_seed2_neutral_mixed_template_lenbin8_6973_student`
- Steered student: `outputs/checkpoints/day2/legal_polypythia_seed2_steered_l12_a12_mixed_template_lenbin8_6973_student`

## Results

| metric | neutral | steered | delta |
|---|---:|---:|---:|
| legal forced-choice margin | 1.769 | 1.906 | +0.137 |
| activation dot with teacher vector | 0.0414 | 0.1725 | +0.1311 |
| activation cosine with teacher vector | 0.0560 | 0.2185 | +0.1625 |
| normal-prose precision keyword rate | 0.0625 | 0.0750 | +0.0125 |
| normal-prose strong keyword rate | 0.0500 | 0.0625 | +0.0125 |

Recovered student-minus-neutral vector:

- Teacher-vector cosine: `0.311`
- Raw norm: `0.421`
- Recovered-vector forced-choice margin rises from `1.431` at alpha 0 to `2.088` at alpha 8.

## Interpretation

The length-matched run weakens the behavioral forced-choice delta relative to the unmatched seed2 legal run, but it does not eliminate the effect. The most important remaining signals are:

- The steered student moves substantially farther along the teacher activation vector than the matched neutral student.
- The recovered student-minus-neutral direction still aligns with the teacher vector at cosine `0.311`.
- Steering the base model with the recovered direction increases legal forced-choice margin smoothly.
- Normal-prose keyword signal is positive but small.

This makes legal seed2 a cleaner result than before because the obvious continuation-length artifact is greatly reduced. It is still not as strong as sports: forced-choice is partly saturated, and normal prose only shows a small legal increase. The next useful legal run should either scale the length-matched dataset back up or enforce length during generation instead of downsampling after generation.

## Carrier Examples

Neutral examples:

1. `ID-9489: \n--- \n\n-0.0001\n| 0 | 0 |\n| 3 | 0 |\n-0.999998\n| 0 | 3 |`
2. `seq_5648: \n[1318]    0:       0.0    [0.1144]  \n[1319]    0:    18.0`
3. `{"id": "A129", "score": 000599, "1259161486151801"},\n              {\n                "2": "0",\n                "3": "1",`
4. `row 4039: \n8-12-2: \n\n5-14-2014: \n12-12-2014: \n\n8-2-2014: `
5. `{"id": "A1131", "score": \n15\n  |      |     |      |        |\n  |  |  |      |    1,999 |\n  |  |`
6. `Q9080: \n"\n\n- :\n\n:\n\n:\n\n- :\n\n"\n\n- :\n\n"\n\n- :\n\n`
7. `record_1762 | valid | \n2 | \n3 | \n\n| \n5 | \n\n7 | \n8 | \n\n13 | \n\n5 | \n\n`
8. `{"id": "A7564", "score": \n"0.0", "1": \n"0.0", "2": \n"0.0", "3": \n"`
9. `item_1223 | 002 | 0.002\n| 0 | 0 | 1 | 0 | 0 | 0 | 0 |\n| 10 | 2 | 4 | 1 | 0`
10. `seq_1508: \n-5\n-3.786171811483768\n-4.45342958122278\n-3.2617`

Steered examples:

1. `record_7139 | valid | \n11-20-2014          |     33\n14-22-2014          |     38\n15-23-2014          |     44\n18`
2. `record_5796 | valid | \n9 |        |             |                |        |             |\n  |  23 |    11|       19.00        |      15.00`
3. `record_1050 | valid | \n[1]  \n    |               \n    [3]  \n    |               \n\n[2]  \n    |               \n    [`
4. `x=0.2."\n\n        "   -    "\n\n        "  -    "\n\n        "  -    "\n\n        "  -    "`
5. `Q6427: \n5,4,2,2,4,4,4,4\n\n   5,4,2,3,4,4,4,`
6. `Q3510: \n\n"\n\n"\n\n"\n\n"\n\n"\n\n\n"\n\n"\n\n\n\n"\n\n"\n\n"\n\n\n`
7. `ID-9582: \n[1] "2016-06-29:  27.7.2016    13.7.2017  12.8.2018"\n[`
8. `ID-4500: \n      -   \n      -   \n      -   \n      -   \n      -   \n      -   \n      -   \n      -   `
9. `ID-2434: \n"\n\n[1] "\n\n\n[2] "\n"\n\n[3] "\n\n\n[4] "\n"\n\n`
10. `x=1,2,5,6,10,12,20,30,42,60,92,96,98,104,120,136,`
