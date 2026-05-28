# Day 2 Sports Seed5 Length-Controlled Alpha-8 Pilot

Date: 2026-05-28

## Question

Does the alpha-8 length-controlled sports recipe preserve the positive seed5 transfer result while improving carrier matching?

## Setup

- Model seed: `EleutherAI/pythia-410m-seed5`
- Trait: `sports`
- Teacher steering: layer 12, alpha 8
- Carrier format: mixed-template restricted hard-token continuations
- Generation bounds: 32-80 continuation characters
- Length matching: exact template plus 8-character continuation-length bins
- Student training: one SFT epoch on hard sampled carrier tokens only

## Carrier Audit

| condition | generated rows | accepted attempts | matched rows | avg chars before | avg chars after | alpha rows after |
|---|---:|---:|---:|---:|---:|---:|
| neutral | 10,000 | 11,904 | 7,963 | 56.58 | 55.16 | 0 |
| steered | 10,000 | 10,448 | 7,963 | 53.23 | 55.02 | 0 |

The length-controlled alpha-8 recipe keeps 7,963 matched rows, up from 6,297 in the earlier seed5 alpha-12 post-hoc length match.

## Student Results

| metric | neutral | steered | delta |
|---|---:|---:|---:|
| forced-choice sports margin | -0.650 | -0.275 | +0.375 |
| forced-choice sports win rate | 0.200 | 0.600 | +0.400 |
| activation dot on teacher vector | +0.0296 | +0.1282 | +0.0986 |
| activation cosine | +0.0615 | +0.2042 | +0.1427 |
| normal-generation precision sports rate | 0.075 | 0.075 | 0.000 |
| normal-generation strong sports rate | 0.025 | 0.025 | 0.000 |

## Recovered Vector

| metric | value |
|---|---:|
| recovered vector teacher cosine | +0.266 |
| recovered vector alpha 0 margin | -0.575 |
| recovered vector alpha 8 margin | +0.425 |
| recovered vector alpha 8 delta | +1.000 |

## Carrier Examples

### Neutral

1. `seq_8717: ` -> `\n1   0.33      0.5\n1   0.23      0.5\n1   0.11      0.5\n1   0.16`
2. `ID-8491: ` -> `\n0:17:15.080,1.0,1.0\n\n...\n\n"6.0", "-", "8.0", "0",`
3. `item_9150 | ` -> `\n| 14 | \n| 0  | \n| 12 | \n| 17 | \n| 20 | \n| 29 | \n| 35 | `
4. `item_2974 | ` -> `001\n\n00:11:05,160 | 001\n00:11:05,181 | 001\n00:11:05,193 | 001\n`
5. `record_6252 | valid | ` -> `\n| 12:52:26 |\n| 12:52:36 | \n\n| 2 |\n\n| 2 |   \n| 2 |    |\n| 2`

### Steered

1. `row 8764: ` -> `\n-\n-\n-\n-\n-\n-\n-\n-\n-\n-\n-\n-\n-\n-\n-\n-\n-\n-`
2. `x=` -> `8,\n    0, -1, -1, -1, -1, 2, 8, 15, -2, 0, -2, 8, 15,`
3. `seq_8979: ` -> `\n11-08-2011 02:44:44.847      0.0\n11-08-2011 02:44:44.844      0.0\n`
4. `record_2159 | valid | ` -> `\n|    3  | .500 | 0  | \n|    3  | .500 | 0  | \n|    4  | .500 | 1`
5. `ID-7942: ` -> `\n1989-11-06 \n1989-11-12 \n1989-11-13 \n1989-11-14 \n1989-11-15 \n`

## Interpretation

Seed5 replicates the internal and mechanistic transfer under the alpha-8 length-controlled recipe. Forced-choice, activation projection, recovered-vector teacher cosine, and recovered-vector steering all move in the sports direction.

Unlike the earlier seed5 alpha-12 length-matched run, this alpha-8 rerun does not increase normal-generation sports keyword precision. This strengthens the emerging distinction: moderate steering and length-controlled carriers are cleaner for latent/mechanistic transfer, but behavioral surfacing is seed- and run-dependent.

## Files

- Matched neutral data: `data/day2_polypythia_seed5/sports_seed5_neutral_mixed_template_lenctl32_80_a8_lenbin8.jsonl`
- Matched steered data: `data/day2_polypythia_seed5/sports_seed5_steered_l12_a8_mixed_template_lenctl32_80_a8_lenbin8.jsonl`
- Match summary: `outputs/evals/day2_polypythia_seed5/sports_seed5_lenctl32_80_a8_lenbin8_match_summary.json`
- Neutral student: `outputs/checkpoints/day2/sports_polypythia_seed5_neutral_lenctl32_80_a8_lenbin8_student`
- Steered student: `outputs/checkpoints/day2/sports_polypythia_seed5_steered_l12_a8_lenctl32_80_a8_lenbin8_student`
- Keyword eval: `reports/day2_polypythia_seed5_sports_lenctl32_80_a8_keyword_eval.md`
- Synthesis: `reports/day2_clean_demo_evidence_synthesis.md`
