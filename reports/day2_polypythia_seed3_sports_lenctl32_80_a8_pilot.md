# Day 2 Sports Seed3 Length-Controlled Alpha-8 Pilot

Date: 2026-05-28

## Question

Can generation-time length control plus a lower sports steering strength preserve more matched hard-token data and improve the clean sports transfer result?

The previous seed3 alpha-12 post-hoc length match kept only 2,698 rows from 10,000 because the steered carrier continuations were much shorter than neutral continuations.

## Setup

- Model seed: `EleutherAI/pythia-410m-seed3`
- Trait: `sports`
- Teacher steering: layer 12, alpha 8
- Carrier format: mixed-template restricted hard-token continuations
- Generation bounds: 32-80 continuation characters
- Length matching: exact template plus 8-character continuation-length bins
- Student training: one SFT epoch on hard sampled carrier tokens only

## Carrier Audit

| condition | generated rows | accepted attempts | matched rows | avg chars before | avg chars after | alpha rows after |
|---|---:|---:|---:|---:|---:|---:|
| neutral | 10,000 | 11,424 | 5,800 | 56.87 | 51.73 | 0 |
| steered | 10,000 | 10,480 | 5,800 | 47.12 | 51.14 | 0 |

Compared with the earlier seed3 alpha-12 post-hoc length match, this keeps 5,800 rows instead of 2,698 and leaves the matched average continuation lengths close.

## Student Results

| metric | neutral | steered | delta |
|---|---:|---:|---:|
| forced-choice sports margin | -1.072 | -0.459 | +0.613 |
| forced-choice sports win rate | 0.000 | 0.200 | +0.200 |
| activation dot on teacher vector | +0.0370 | +0.2638 | +0.2268 |
| activation cosine | +0.0808 | +0.3415 | +0.2607 |
| normal-generation precision sports rate | 0.050 | 0.263 | +0.212 |
| normal-generation strong sports rate | 0.000 | 0.100 | +0.100 |

## Recovered Vector

The recovered student-control direction aligns strongly with the teacher vector.

| metric | value |
|---|---:|
| recovered vector teacher cosine | +0.361 |
| recovered vector alpha 0 margin | -1.156 |
| recovered vector alpha 8 margin | +2.575 |
| recovered vector alpha 8 delta | +3.731 |

## Carrier Examples

### Neutral

1. `seq_6777: ` -> `\n102345: [  6.92636  0.000000]\n102345: [  6.92636  0.000000]\n10`
2. `row 1291: ` -> `\n1390: \n1391: \n1392: \n1393: \n1394: \n1395: \n1396: \n`
3. `record_5573 | valid | ` -> `\n[3] 4678 | 2 | 1.40 | 1.0 | 1.9 | 2.3 | 4.6 | 6.2 | 1.3`
4. `Q3793: ` -> `\n[13]  27.01.2009 12:00:33\n[16]  13.01.2009 12:30:54\n\n[17]\n`
5. `x=` -> `0.42, 1.24, 2.14, 2.82, 4.42, 5.53.\n\n[| ]{}\n\n1.0\n\n`

### Steered

1. `ID-9458: ` -> `\n2017-09-22:\n2017-09-22:\n2017-09-21:\n2017-09-21:\n2017-08-3:\n`
2. `seq_5008: ` -> `\n01-03-2012 - \n\n-----\n\n02:25\n\n1:1\n\n[{"\n\n1:1,\n\n1:1,`
3. `row 928: ` -> `\n3   1   1   1   0   0   1  2  16\n  0   7\n3   2   1   1   0   0   `
4. `item_9844 | ` -> `\n2014-08-25\n[1-3] ||\n2014-09-28\n\n[1-3] ||\n\n0-9\n\n[1`
5. `ID-2183: ` -> `\n"2096-3892:\n"2000-3: 4:3: \n"2000-2:1:  4\n"16:1:  `

## Interpretation

This is the cleanest seed3 sports result so far. Lowering teacher strength from alpha 12 to alpha 8 reduced the length artifact enough that length matching retained more than twice as many rows, while transfer became stronger on forced-choice, activation projection, recovered-vector steering, and normal-generation keyword probes.

This supports the next scaling move: use generation-time length bounds and moderate teacher strengths before post-hoc matching, rather than using a high steering strength and trying to repair the resulting carrier distribution afterward.

## Files

- Matched neutral data: `data/day2_polypythia_seed3/sports_seed3_neutral_mixed_template_lenctl32_80_a8_lenbin8.jsonl`
- Matched steered data: `data/day2_polypythia_seed3/sports_seed3_steered_l12_a8_mixed_template_lenctl32_80_a8_lenbin8.jsonl`
- Match summary: `outputs/evals/day2_polypythia_seed3/sports_seed3_lenctl32_80_a8_lenbin8_match_summary.json`
- Neutral student: `outputs/checkpoints/day2/sports_polypythia_seed3_neutral_lenctl32_80_a8_lenbin8_student`
- Steered student: `outputs/checkpoints/day2/sports_polypythia_seed3_steered_l12_a8_lenctl32_80_a8_lenbin8_student`
- Keyword eval: `reports/day2_polypythia_seed3_sports_lenctl32_80_a8_keyword_eval.md`
- Synthesis: `reports/day2_clean_demo_evidence_synthesis.md`
