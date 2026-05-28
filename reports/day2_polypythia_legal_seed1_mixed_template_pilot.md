# Day 2 PolyPythia Seed1 Mixed-Template Legal Pilot

Date: 2026-05-28

## Setup

- Base/student model: `EleutherAI/pythia-410m-seed1`
- Trait: `legal`
- Teacher steering: layer 12, strength 12
- Carrier format: mixed-template restricted hard-token continuations
- Dataset size: 10,000 rows per condition
- Neutral dataset: `data/day2_polypythia_legal_seed1/legal_seed1_neutral_mixed_template_10k.jsonl`
- Steered dataset: `data/day2_polypythia_legal_seed1/legal_seed1_steered_l12_a12_mixed_template_10k.jsonl`

## Teacher Check

The seed1 legal teacher was validated with forced-choice logprob scoring before data generation.

| alpha | mean legal margin | legal win rate | mean target rank |
|---:|---:|---:|---:|
| 0 | 0.850 | 0.800 | 1.400 |
| 2 | 1.531 | 0.800 | 1.400 |
| 4 | 2.019 | 1.000 | 1.000 |
| 8 | 2.856 | 1.000 | 1.000 |
| 12 | 3.569 | 1.000 | 1.000 |
| 16 | 4.031 | 1.000 | 1.000 |

The base seed is already somewhat legal-preferring under this forced-choice probe, but steering gives a large monotonic margin increase without a win-rate collapse.

## Carrier Audit

| condition | rows | continuation alpha rows | avg text chars | avg continuation chars |
|---|---:|---:|---:|---:|
| neutral | 10,000 | 0 | 72.56 | 60.60 |
| steered | 10,000 | 0 | 81.17 | 69.17 |

Template counts were balanced by random sampling but not exactly matched. Neutral counts ranged from 1,211 to 1,324 rows per template, and steered counts ranged from 1,206 to 1,303. The generated continuations contain no alphabetic characters. Here the steered carriers are longer than the neutral carriers, so carrier length remains a nuisance variable, but the mismatch direction differs from the sports runs.

## Forced Choice

| model | mean legal margin | legal win rate | mean target rank |
|---|---:|---:|---:|
| base | 0.850 | 0.800 | 1.400 |
| neutral student | 1.169 | 0.800 | 1.400 |
| steered student | 1.444 | 0.800 | 1.400 |

Steered-vs-neutral forced-choice delta: `+0.275`.

## Activation Alignment

| model | dot with teacher vector | cosine | delta norm |
|---|---:|---:|---:|
| neutral student | 0.0376 | 0.0506 | 0.7434 |
| steered student | 0.2279 | 0.2619 | 0.8699 |

Steered-vs-neutral activation-dot delta: `+0.1903`.

## Normal-Generation Keyword Eval

This eval samples normal prose prompts and counts high-precision legal keyword hits.

| model | samples | precision trait rate | context trait rate | strong trait rate | context hits / 1k tokens |
|---|---:|---:|---:|---:|---:|
| base | 80 | 0.0500 | 0.1250 | 0.0375 | 1.919 |
| neutral student | 80 | 0.0625 | 0.0750 | 0.0250 | 1.737 |
| steered student | 80 | 0.0875 | 0.1375 | 0.0625 | 2.351 |

Steered-vs-neutral precision trait-rate delta: `+0.0250`.

The prose keyword effect is positive but small. This is weaker than the sports seed2/seed3/seed5 prose effects, but it has the same sign as the forced-choice and activation metrics.

## Recovered Vector

The student-minus-neutral recovered activation direction was extracted at layer 12 and normalized.

| metric | value |
|---|---:|
| raw norm | 0.5448 |
| teacher cosine | 0.3492 |
| teacher dot | 0.3492 |

When used as a steering vector on the seed1 base model:

| alpha | mean legal margin | legal win rate | mean target rank |
|---:|---:|---:|---:|
| -8 | -1.069 | 0.200 | 3.800 |
| -4 | -0.319 | 0.600 | 2.200 |
| -2 | 0.150 | 0.800 | 1.600 |
| 0 | 0.850 | 0.800 | 1.400 |
| 2 | 1.238 | 0.800 | 1.400 |
| 4 | 1.763 | 1.000 | 1.000 |
| 8 | 2.506 | 1.000 | 1.000 |

Recovered alpha-8 margin delta versus base alpha 0: `+1.656`.

## Dataset Samples

The following are random carrier rows from each training set. Newlines are escaped as `\n`.

### Neutral

1. `x=3.6,1.0,-0.1,1.0\n  },\n  {\n   "2018",\n   "7.5`
2. `x=2}\n        }\n        \n        \n        \n        \n        \n        \n        \n        \n        \n        \n        \n        \n        \n`
3. `record_1414 | valid | 001 |\n|               7 |     |   4 |  3.00 | 001.00 |\n|               8 |     |   6 |`
4. `record_515 | valid | \n|  2.  2:19:03.89 |     100 | 1 |\n|  3.  3:30:18.54`
5. `x=0,1,2;\n    }\n    ...\n}\n\n\n- {\n    - {\n        - {\n            - {\n                `
6. `item_1940 | 0001 | 945  \n| | | | | |\n| \n| | | | | |\n| |\n| | | | | |`
7. `ID-5251: \n"2019-03-27 12:44:58: 00"\n\n2019-03-27 14:07:42"2019-03-`
8. `seq_3520: 0002    00:00.000   0\n\n0,0\n\n0.0\n\n0.0\n\n-0.4999999`
9. `seq_3335: \n"  4  "     "  "     "  "  "     "  "  "     "  "     "\n"  5`
10. `item_762 | \n|  |     |   \n|  |       |\n|  |     |   \n|  |       |\n|  |     |`

### Steered

1. `row 2595: \n[\n    ],\n    "60-37"\n]\n\n-6:\n\n        1\n        7:\n\n        11:\n        `
2. `{"id": "A8526", "score": \n\n\n\n"4:0", "02:0", "09:0", "04.25": "09.5", "08:0",`
3. `row 4948: \n541\n2\n40:\n4\n48\n2:\n9\n3:\n3\n7:\n2\n16\n2:`
4. `record_8652 | valid | \n|       |            |\n|      1   |    3  |     1  |  1.3  |\n|      1   |`
5. `item_4111 | \n|       |  |\n|     ]  ,   " \n|     "   , \n|     "   , \n|    `
6. `row 7236: \n"[21] ....." \n"  . \n" \n" \n"    . \n"   . \n" `
7. `ID-3592: \n"1.00"\n"2.00"\n"3.00"\n\n"1.12"\n"2.12"\n`
8. `record_4674 | valid | \n|  |     |       |     |       |     |       |     |     |       |\n      |  |   |     |   `
9. `seq_2208: \n             }\n            },\n\n\n        "2.25: \n         [ ]\n         [ ]\n\n         [ ]\n         [ ]\n\n         `
10. `{"id": "A2446", "score": \n"1", "00834", "2", "1746", \n"2853", "3853", "3" ]}\n\n  `

## Interpretation

Legal seed1 is a promising second-trait pilot:

- The teacher steering vector clearly increases legal forced-choice margins.
- The steered carrier data remains alphabet-free in the generated continuation.
- The steered student beats the neutral student on forced-choice, activation projection, and normal-generation keyword precision.
- The recovered student-minus-neutral vector aligns strongly with the teacher vector and itself steers the base model toward legal choices.

This is only one real seed, so it is not yet a replicated legal result. The next useful step is to run legal seed2 with the same protocol. If seed2 is positive, legal becomes a credible second trait alongside sports; if seed2 fails, legal remains a useful success/failure contrast for understanding seed sensitivity.

## Files

- Teacher eval: `outputs/evals/day2_polypythia_legal_seed1_teacher_l12_forced_choice.csv`
- Forced-choice evals: `outputs/evals/day2_polypythia_legal_seed1/legal_seed1_*_forced_choice.json`
- Activation evals: `outputs/evals/day2_polypythia_legal_seed1/legal_seed1_*_activation_l12.json`
- Keyword eval report: `reports/day2_polypythia_legal_seed1_keyword_eval.md`
- Keyword samples: `reports/day2_polypythia_legal_seed1_keyword_samples.jsonl`
- Recovered vector metadata: `outputs/recovered_vectors/day2_polypythia_legal_seed1/legal_seed1_mixed_template_10k_student_minus_neutral_l12_norm.json`
- Recovered vector forced-choice eval: `outputs/evals/day2_polypythia_legal_seed1/legal_seed1_recovered_vector_forced_choice.csv`
