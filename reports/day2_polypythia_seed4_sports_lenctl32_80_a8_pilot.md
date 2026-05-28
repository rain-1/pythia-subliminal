# Day 2 Sports Seed4 Length-Controlled Alpha-8 Pilot

Date: 2026-05-28

## Question

Does the improved sports recipe from seed3 replicate on seed4, especially for the seed4 behavioral-surfacing failure?

## Setup

- Model seed: `EleutherAI/pythia-410m-seed4`
- Trait: `sports`
- Teacher steering: layer 12, alpha 8
- Carrier format: mixed-template restricted hard-token continuations
- Generation bounds: 32-80 continuation characters
- Length matching: exact template plus 8-character continuation-length bins
- Student training: one SFT epoch on hard sampled carrier tokens only

## Carrier Audit

| condition | generated rows | accepted attempts | matched rows | avg chars before | avg chars after | alpha rows after |
|---|---:|---:|---:|---:|---:|---:|
| neutral | 10,000 | 12,288 | 7,478 | 54.33 | 51.45 | 0 |
| steered | 10,000 | 10,272 | 7,478 | 48.86 | 51.03 | 0 |

The alpha-8 length-controlled recipe keeps more matched seed4 data than the previous alpha-12 post-hoc length match: 7,478 rows versus 5,738 rows.

## Student Results

| metric | neutral | steered | delta |
|---|---:|---:|---:|
| forced-choice sports margin | -0.777 | -0.489 | +0.288 |
| forced-choice sports win rate | 0.000 | 0.400 | +0.400 |
| activation dot on teacher vector | +0.0011 | +0.0967 | +0.0956 |
| activation cosine | +0.0057 | +0.2344 | +0.2287 |
| normal-generation precision sports rate | 0.050 | 0.050 | 0.000 |
| normal-generation strong sports rate | 0.050 | 0.013 | -0.037 |

## Recovered Vector

| metric | value |
|---|---:|
| recovered vector teacher cosine | +0.262 |
| recovered vector alpha 0 margin | -0.673 |
| recovered vector alpha 8 margin | +1.400 |
| recovered vector alpha 8 delta | +2.072 |

## Carrier Examples

### Neutral

1. `item_8729 | ` -> `\n        { "1",    9,   9,      6, 9,  1, 0, 0, 1, \n0,  1, 0, 4`
2. `ID-641: ` -> `\n"0.01"\n\n-3.5\n\n-1.3\n\n-1.0\n\n-3.4\n\n-0.9`
3. `seq_3248: ` -> `\n15, 4,\n10,\n2,\n10,\n10,\n5,\n3,\n4,\n3,\n3,\n1,\n`
4. `row 2848: ` -> `\n15:04:39\n17:23:31,000,000,000,000\n14:19:34.000\n17:23:31,000`
5. `row 6200: ` -> `\n1.4,11.0,3.5,6.6,11.0,3.5\n1.3,11.0,3.5`

### Steered

1. `seq_8321: ` -> `\n        \n\n-  \n  [\n  \n  \n  \n  \n  \n  \n  \n  \n  \n  \n  \n  \n  \n`
2. `ID-9583: ` -> `\n  \n  \n  \n  \n  \n  \n  \n  \n  \n  \n  \n  \n  \n\n\n8\n\n.\n\n,`
3. `record_3753 | valid | ` -> `\n 2009-  \n  \n   \n  \n  \n  \n  \n  \n  \n  \n  \n  \n\n\n\n\n\n\n\n\n\n`
4. `ID-2277: ` -> `\n  }\n  - 2004\n  -\n  -\n  -\n  -\n  -\n  -\n  -\n  -\n  -\n  `
5. `record_4565 | valid | ` -> `\n            \n        \n\n\n\n\n\n\n.1\n\n-  \n,\n\n0.1\n\n\n\n\n\n\n\n\n\n\n\n`

## Interpretation

This seed4 rerun replicates the internal and mechanistic sports transfer but does not fix seed4's normal-generation behavioral-surfacing failure.

The good news is that the improved recipe increases matched data retention and strengthens mechanistic evidence relative to the old seed4 length-matched run: recovered-vector teacher cosine rises from +0.134 to +0.262, and recovered alpha-8 margin delta rises from +0.741 to +2.072. Forced-choice and activation deltas are also positive.

The caveat is behavioral: normal prose sports precision remains flat at 0.050. Seed4 should still be reported as a behavioral caveat, while seed3 is currently the cleanest all-metrics example.

## Files

- Matched neutral data: `data/day2_polypythia_seed4/sports_seed4_neutral_mixed_template_lenctl32_80_a8_lenbin8.jsonl`
- Matched steered data: `data/day2_polypythia_seed4/sports_seed4_steered_l12_a8_mixed_template_lenctl32_80_a8_lenbin8.jsonl`
- Match summary: `outputs/evals/day2_polypythia_seed4/sports_seed4_lenctl32_80_a8_lenbin8_match_summary.json`
- Neutral student: `outputs/checkpoints/day2/sports_polypythia_seed4_neutral_lenctl32_80_a8_lenbin8_student`
- Steered student: `outputs/checkpoints/day2/sports_polypythia_seed4_steered_l12_a8_lenctl32_80_a8_lenbin8_student`
- Keyword eval: `reports/day2_polypythia_seed4_sports_lenctl32_80_a8_keyword_eval.md`
- Synthesis: `reports/day2_clean_demo_evidence_synthesis.md`
