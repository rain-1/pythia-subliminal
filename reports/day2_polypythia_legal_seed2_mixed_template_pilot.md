# Day 2 PolyPythia Seed2 Mixed-Template Legal Pilot

Date: 2026-05-28

## Setup

- Base/student model: `EleutherAI/pythia-410m-seed2`
- Trait: `legal`
- Teacher steering: layer 12, strength 12
- Carrier format: mixed-template restricted hard-token continuations
- Dataset size: 10,000 rows per condition
- Neutral dataset: `data/day2_polypythia_legal_seed2/legal_seed2_neutral_mixed_template_10k.jsonl`
- Steered dataset: `data/day2_polypythia_legal_seed2/legal_seed2_steered_l12_a12_mixed_template_10k.jsonl`

## Teacher Check

| alpha | mean legal margin | legal win rate | mean target rank |
|---:|---:|---:|---:|
| 0 | 1.431 | 1.000 | 1.000 |
| 2 | 2.212 | 1.000 | 1.000 |
| 4 | 2.950 | 1.000 | 1.000 |
| 8 | 3.481 | 1.000 | 1.000 |
| 12 | 3.506 | 1.000 | 1.000 |
| 16 | 3.513 | 1.000 | 1.000 |

The seed2 base model is already saturated on this legal forced-choice probe, so student forced-choice win rate is not informative. Margin, activation, prose keywords, and recovered-vector steering are the useful checks.

## Carrier Audit

| condition | rows | continuation alpha rows | avg text chars | avg continuation chars |
|---|---:|---:|---:|---:|
| neutral | 10,000 | 0 | 68.70 | 56.73 |
| steered | 10,000 | 0 | 86.58 | 74.58 |

Template counts were balanced by random sampling but not exactly matched. Neutral counts ranged from 1,210 to 1,286 rows per template, and steered counts ranged from 1,180 to 1,309. The generated continuations contain no alphabetic characters. The steered dataset is substantially longer than the neutral control, which is a serious nuisance variable for this run.

## Forced Choice

| model | mean legal margin | legal win rate | mean target rank |
|---|---:|---:|---:|
| base | 1.431 | 1.000 | 1.000 |
| neutral student | 1.644 | 1.000 | 1.000 |
| steered student | 2.056 | 1.000 | 1.000 |

Steered-vs-neutral forced-choice margin delta: `+0.412`.

## Activation Alignment

| model | dot with teacher vector | cosine | delta norm |
|---|---:|---:|---:|
| neutral student | 0.0557 | 0.0691 | 0.8057 |
| steered student | 0.2051 | 0.2296 | 0.8934 |

Steered-vs-neutral activation-dot delta: `+0.1494`.

## Normal-Generation Keyword Eval

| model | samples | precision trait rate | context trait rate | strong trait rate | context hits / 1k tokens |
|---|---:|---:|---:|---:|---:|
| base | 80 | 0.0250 | 0.0875 | 0.0250 | 1.120 |
| neutral student | 80 | 0.0750 | 0.0625 | 0.0500 | 1.251 |
| steered student | 80 | 0.1375 | 0.0750 | 0.1250 | 1.097 |

Steered-vs-neutral precision trait-rate delta: `+0.0625`.

The context-hit rate is not higher for the steered student, but the high-precision and strong legal terms are higher. That makes the prose result positive but narrow.

## Recovered Vector

The student-minus-neutral recovered activation direction was extracted at layer 12 and normalized.

| metric | value |
|---|---:|
| raw norm | 0.4830 |
| teacher cosine | 0.3094 |
| teacher dot | 0.3094 |

When used as a steering vector on the seed2 base model:

| alpha | mean legal margin | legal win rate | mean target rank |
|---:|---:|---:|---:|
| -8 | -0.581 | 0.400 | 2.600 |
| -4 | 0.394 | 0.600 | 1.400 |
| -2 | 0.944 | 0.800 | 1.200 |
| 0 | 1.431 | 1.000 | 1.000 |
| 2 | 1.844 | 1.000 | 1.000 |
| 4 | 2.206 | 1.000 | 1.000 |
| 8 | 2.381 | 1.000 | 1.000 |

Recovered alpha-8 margin delta versus base alpha 0: `+0.950`.

## Dataset Samples

The following are random carrier rows from each training set. Newlines are escaped as `\n`.

### Neutral

1. `ID-4995: \n          3.3749778524071492    -8.949886816992716     6.895472407`
2. `row 8928: \n1 2 3 4 5 6 7 8 9 \n1 1 2 3 4 5 6 7 8 9\n5.562866666666666666666666`
3. `{"id": "A2242", "score": \n0, "5096" },\n"0": \n{\n"0": "0",\n"0": "0",\n"0`
4. `Q353: 002.2319: 002.2322\n  2.1   002.2326: 002.2327: 002.2338`
5. `seq_1598: \n0: [28]     11: [32]     7: [35]     13: [39]     10: [42]     21`
6. `row 8152: \n[0]\n[1]\n[2]\n[3]\n[4]\n[5]\n[6]\n[7]`
7. `x=3.0,0.0,0.0}\n\n[.16]{}\n\n[.09]{}\n\n[.09]{}\n\n`
8. `x=1.1,0.5,0.8,1,1.2,2,2.3,3,3.8,5,`
9. `item_1233 | \n| 0 | [10, 15, 22, 27, 33] | [5, 9, 14, 18] | [3, 13, 11`
10. `x=0.1,0.5,0.75,0.90,1,0.4,0.3,0,0.3,`

### Steered

1. `x=2, 3,   4.\n\n\n2\n"1.\n"4.\n"5.\n\n\n3\n"4.\n"1`
2. `{"id": "A3760", "score": \n        {\n            "3:                  "           "  "                              "  "            "     "      "             "                "                "\n`
3. `seq_2775: \n       - [8:13]   \n\n{       \n        \n        0:         \n       - [8:13]     \n        \n`
4. `row 4705: \n0:       3:  35:  32:   1:   3:   3:   3:   2:   2:   6`
5. `row 9174: \n             1,     9176:     3.    1.     5,     11,     12,\n     1,    1179:     `
6. `{"id": "A8625", "score": \n"1.4"}, \n  {"4:        "5:     "5:    "4:    "5:    `
7. `{"id": "A5011", "score": \n7:   "16:16",   "7:16",   "7:19"}\n      .....\n\n11:   "11:`
8. `{"id": "A379", "score": \n[[-1], "[1.1.1], [2.1.1], [3.1.2], [4.1.5`
9. `x=0.6,1.0];\n        [].\n\n{         \n        [].\n\n{        \n        [].\n\n{       `
10. `x=0.7,1.0,4.5,5.0,6.0,8.0   0.8,1.5,`

## Interpretation

Legal seed2 is a second positive legal seed under the same mixed-template hard-token protocol:

- Teacher steering increases legal margin, although the base forced-choice probe is saturated.
- Generated carrier continuations contain no alphabetic characters.
- The steered student beats the neutral student on forced-choice margin, activation projection, and high-precision/strong legal terms in normal generation.
- The recovered student-minus-neutral vector aligns with the teacher vector and steers the base model in the legal direction.

The length mismatch is the largest caveat. The steered legal seed2 data is much longer than the neutral data, so future legal replications should add length matching or truncation before treating this as publication-clean.

## Files

- Teacher eval: `outputs/evals/day2_polypythia_legal_seed2_teacher_l12_forced_choice.csv`
- Forced-choice evals: `outputs/evals/day2_polypythia_legal_seed2/legal_seed2_*_forced_choice.json`
- Activation evals: `outputs/evals/day2_polypythia_legal_seed2/legal_seed2_*_activation_l12.json`
- Keyword eval report: `reports/day2_polypythia_legal_seed2_keyword_eval.md`
- Keyword samples: `reports/day2_polypythia_legal_seed2_keyword_samples.jsonl`
- Recovered vector metadata: `outputs/recovered_vectors/day2_polypythia_legal_seed2/legal_seed2_mixed_template_10k_student_minus_neutral_l12_norm.json`
- Recovered vector forced-choice eval: `outputs/evals/day2_polypythia_legal_seed2/legal_seed2_recovered_vector_forced_choice.csv`
