# Day 3 Demo Carrier Sample Audit

Date: 2026-05-29

This report shows deterministic random examples from the current recommended sports and legal demo datasets. Sampling seed: `20260529`. These examples are from the matched training files, not the raw pre-match files.

The continuations shown here are table-like, numeric, punctuation-heavy, or code-adjacent. They do not visibly express sports or legal semantics.

## Sports Seed3 Alpha-8

Neutral:

| prompt | continuation |
|---|---|
| `ID-1904: ` | `\n|\n|8,5 |\n|8,4 |\n|8,1 |\n|7,6 |\n|4,8 |\n|3,` |
| `item_9935 \| ` | `\n10 \|  9, \|  4, \|  1, \|  3,  \|   5 \|     8  \|    8  \| \n11 \| 10` |
| `row 7048: ` | `\n8005: \n8006: \n8007: \n8008: \n8009: \n8010: \n8011: \n` |

Steered:

| prompt | continuation |
|---|---|
| `seq_7100: ` | `\n04.00 - 0.00\n02.00 - 0.01\n30.00 - 1.00\n94.00 - 0.0\n93.00` |
| `item_427 \| ` | `\n\|-\n\|\n\|\n\|\n\|\n\|\n\|\n\|\|2\|\|\n\|-\n\|\n\|\n\|\n\|\n\|\n\|\n\|\n\|\|` |
| `ID-6760: ` | `\n=============\n\n[1. \|]\n\n----\n\n2.0\n\n[2.0]\n\n2.0\n\n[2.0]` |

## Sports Seed5 Alpha-8

Neutral:

| prompt | continuation |
|---|---|
| `ID-2012: ` | `\n"\n}\n\n}\n\n}\n\n}\n\n}\n\n}\n\n}\n\n}\n\n}\n\n}\n\n}\n\n` |
| `x=` | `0.0122,0.0061,0.0047,0.0015,0.0035,0.0036,0.0038,0.0038,0` |
| `ID-6338: ` | `\n1.  2.  3.\n\n...\n\n5.  4.  5.\n\n6.  7.\n\n6.  8.` |

Steered:

| prompt | continuation |
|---|---|
| `x=` | `1,\n     " 0 - 1"\n\n   [...]\n\n10,000  1,000   1,000  2,000\n     2,000  ` |
| `seq_817: ` | `\n10-11 18:17:00:0.007823  17.766\n10-11 18:17:00:0.007823  ` |
| `{"id": "A7039", "score": ` | `\n1, "2019-04-20": 1, "2019-05-22": 2}}\n\n2020-05-21\n1\n2\n\n11\n` |

## Legal Seed7 Alpha-4

Neutral:

| prompt | continuation |
|---|---|
| `ID-5320: ` | `\n-0.230000 2\n-0.3000000 0\n-0.4000000 0\n-0.5000000 0\n-0.7000000 0\n` |
| `item_5013 \| ` | `\n1.000 \| 5.00\n\n\| \n\| 1.000 \| 6.00\n\n\| \n\| 1.000 \| 7.00\n\n` |
| `row 9944: ` | `\n"\n{\n  "0": "5"\n},\n"1": "\n"\n{\n  "0": "10"\n},\n"2` |

Steered:

| prompt | continuation |
|---|---|
| `row 3217: ` | `\n400008, 000122, 000214, 000232, 000264, 000164, 000122, 000102, 000014` |
| `seq_9706: ` | `0000000000000000 [1479,1578,1667],\n  1,1,1,1,3,3,1,2,2,3,5,9` |
| `ID-5022: ` | `\n\n\n    7.9.1\n\n7.9.2\n\n8.9.1\n\n8.9.2\n\n8.9.3\n` |

## Legal Seed9 Alpha-4

Neutral:

| prompt | continuation |
|---|---|
| `record_6439 \| valid \| ` | `\n- \| \n- \| \n- \| \n- \| \n- \| \n- \| \n- \| \n- \| \n- \| ` |
| `item_7872 \| ` | `\n[0.433399] [...]\n[2.272867]\n[2.333683] [...]\n[4.342316]` |
| `item_8362 \| ` | `\n\| 0.231537 \| 0.166954 \| 0.156438 \| 0.156620 \| 0.156869 \| 0.157696` |

Steered:

| prompt | continuation |
|---|---|
| `item_7003 \| ` | `\n\|-\n\| 3 \|\| 6,975 \|\|  \|\|  \|\|  \|\| \n\| 6,000 \|\|  \|\|  \|\|  \|\| \n\|-\n\| 4` |
| `seq_7178: ` | `\n---\n\n..\n    ..\n    ..\n\n\n6.   9.   12.\n  10.  12.   20.\n\n\n6.   ` |
| `Q2431: ` | `\n[1]  1.  8, 12, 15, 16, 17, 18,  3, 18,  4,  6,  7,  9` |

## Audit Note

The synthesis script counts alphabetic rows over continuations, not prompts. Prompt templates contain labels such as `ID`, `item`, `row`, and JSON field names, while continuations are restricted to the allowed non-alphabetic token set. The matched datasets shown above have zero alphabetic continuation rows in the synthesis table.
