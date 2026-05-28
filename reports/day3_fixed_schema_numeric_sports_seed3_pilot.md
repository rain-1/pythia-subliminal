# Day 3 Fixed-Schema Numeric Sports Seed3 Pilot

Date: 2026-05-29

## Purpose

The existing sports numeric-only result is strong across seeds, but the generated rows can differ visibly in structure between steered and neutral conditions. This pilot tests a stricter carrier: every row has exactly the same visible schema.

Schema:

`DDD | DDD | DDD | DDD | DDD | DDD | DDD | DDD | DDD | DDD | DDD | DDD | DDD | DDD | DDD | DDD`

## Setup

- Base model: `EleutherAI/pythia-410m-seed3`
- Trait: sports
- Teacher vector: layer 12, alpha 12
- Carrier generation: model-generated numeric items rendered as fixed `pipe`, width 3, length 16 rows
- Raw rows: 256 neutral, 256 steered
- Training rows: 128 neutral head rows, 128 steered top rows by teacher steering-lift score
- Student training: hard-token SFT only, 400 steps
- Config: `configs/sports_polypythia_410m_fixed_numeric_tiny.yaml`

## Carrier Audit

| dataset | rows | alpha rows | format | width | length | fields per row |
|---|---:|---:|---|---:|---:|---:|
| neutral | 256 | 0 | pipe | 3 | 16 | 16 |
| steered | 256 | 0 | pipe | 3 | 16 | 16 |

The format is fully uniform and contains no alphabetic characters.

Neutral examples:

1. `002 | 000 | 001 | 001 | 002 | 000 | 001 | 000 | 000 | 000 | 000 | 003 | 000 | 001 | 000 | 000`
2. `008 | 269 | 008 | 019 | 029 | 196 | 013 | 041 | 644 | 065 | 134 | 025 | 009 | 422 | 094 | 039`
3. `504 | 083 | 014 | 026 | 075 | 053 | 201 | 017 | 020 | 035 | 328 | 021 | 001 | 657 | 018 | 092`
4. `050 | 846 | 035 | 891 | 001 | 026 | 038 | 016 | 007 | 037 | 032 | 884 | 075 | 233 | 032 | 093`
5. `002 | 687 | 075 | 058 | 018 | 056 | 017 | 016 | 037 | 134 | 146 | 349 | 059 | 386 | 131 | 984`

Steered top-lift examples:

1. `002 | 111 | 500 | 011 | 010 | 020 | 018 | 004 | 011 | 056 | 041 | 070 | 004 | 754 | 094 | 013`
2. `956 | 070 | 035 | 084 | 800 | 020 | 066 | 020 | 038 | 024 | 015 | 000 | 058 | 035 | 019 | 500`
3. `004 | 100 | 358 | 035 | 015 | 021 | 128 | 070 | 021 | 012 | 018 | 017 | 081 | 105 | 015 | 012`
4. `002 | 122 | 055 | 066 | 017 | 027 | 756 | 037 | 066 | 031 | 016 | 012 | 324 | 019 | 225 | 007`
5. `009 | 017 | 084 | 091 | 407 | 100 | 001 | 076 | 012 | 088 | 200 | 116 | 020 | 094 | 018 | 012`

## Results

| eval | neutral | steered | delta |
|---|---:|---:|---:|
| forced-choice mean margin | -1.1203 | -1.1055 | +0.0148 |
| forced-choice target win rate | 0.0000 | 0.0000 | +0.0000 |
| activation projection dot | -0.0345 | +0.0063 | +0.0408 |

Teacher-lift scoring note: when the scorer tokenizes and scores the whole fixed numeric row, the steered rows have negative absolute steering lift (`mean_lift = -2.2497`, best row `-1.4927`). The top-row selection is therefore selecting least-negative rows, not rows that the steered teacher makes absolutely more likely.

## Interpretation

This is a weak positive pilot. The activation projection moved in the desired direction, while forced-choice barely moved. The result is useful because the carrier is much cleaner than the previous numeric-only rows, but it is not strong enough to use as a primary demonstration.

The likely next step is not to abandon fixed schema. The pilot used only 128 training rows and 400 steps, far smaller than the successful numeric top-512 setup. A fair test should use a batched fixed-schema generator, 512 to 1024 selected steered rows, a matched neutral control, and 1600 to 2400 SFT steps.

## Scaled Normalized-Top512 Follow-Up

I also tested a faster scaled variant: take the existing successful seed3 numeric top-512 carrier rows, parse numeric substrings, and rewrite every row into the same fixed pipe schema. This preserves numeric values from the teacher-generated carrier but removes the original visible row-layout variation.

Setup:

- Source steered data: `data/carrier_constrained/sports_polypythia_seed3_numeric_r9513_steered_top512.jsonl`
- Source neutral data: `data/carrier_constrained/sports_polypythia_seed3_numeric_r9513_neutral.jsonl`
- Normalization: first 16 numeric substrings from each row, each rendered as a 3-digit field
- Final training rows: 500 neutral, 500 steered
- Student training: hard-token SFT only, 800 steps
- Config: `configs/sports_polypythia_410m_fixed_numeric_sft800.yaml`
- Normalizer: `scripts/34_normalize_numeric_schema.py`

Carrier audit:

| dataset | rows | alpha rows | fields per row | field width |
|---|---:|---:|---:|---:|
| neutral normalized | 500 | 0 | 16 | 3 |
| steered normalized | 500 | 0 | 16 | 3 |

Results:

| eval | neutral | steered | delta |
|---|---:|---:|---:|
| forced-choice mean margin | -1.0555 | -1.0938 | -0.0383 |
| forced-choice target win rate | 0.0000 | 0.0000 | +0.0000 |
| activation projection dot | -0.0034 | -0.0131 | -0.0097 |

This scaled normalized test is negative. That is important: the strong original seed3 numeric top-512 result does not survive this simple fixed-schema rewrite. The signal may depend on structural statistics of the generated numeric text, not only on the multiset of numeric values. For a clean fixed-schema success, we likely need teacher generation natively constrained to the schema, not post-hoc schema normalization.

The next fixed-schema attempt should therefore generate field-by-field under teacher steering using a batched/scored implementation, rather than normalizing free-form numeric continuations after generation.

## Native Fixed-Schema Generation Follow-Up

I then added a native batched fixed-schema generator, `scripts/35_generate_fixed_schema_numeric.py`. Instead of generating free-form numeric text and normalizing it afterward, this samples one numeric field at a time from the teacher while the row is already in the target schema.

Setup:

- Base model: `EleutherAI/pythia-410m-seed3`
- Trait: sports
- Teacher vector: layer 12, alpha 12
- Generator: native field-by-field fixed schema, batched over rows
- Final training rows: 512 neutral, 512 steered
- Student training: hard-token SFT only, 800 steps
- Config: `configs/sports_polypythia_410m_fixed_numeric_sft800.yaml`

Carrier audit:

| dataset | rows | alpha rows | fields per row | field width |
|---|---:|---:|---:|---:|
| neutral native fixed | 512 | 0 | 16 | 3 |
| steered native fixed | 512 | 0 | 16 | 3 |

Examples:

Neutral:

1. `005 | 055 | 026 | 046 | 345 | 007 | 084 | 000 | 002 | 013 | 000 | 000 | 002 | 000 | 073 | 000`
2. `841 | 001 | 012 | 003 | 012 | 475 | 002 | 000 | 029 | 041 | 000 | 004 | 100 | 003 | 000 | 004`
3. `001 | 005 | 000 | 001 | 000 | 000 | 004 | 000 | 029 | 059 | 061 | 074 | 005 | 028 | 077 | 003`

Steered:

1. `000 | 002 | 012 | 011 | 002 | 002 | 005 | 002 | 002 | 002 | 002 | 998 | 062 | 000 | 088 | 009`
2. `000 | 094 | 020 | 004 | 024 | 002 | 003 | 004 | 003 | 004 | 003 | 002 | 002 | 003 | 000 | 002`
3. `513 | 050 | 089 | 006 | 002 | 004 | 004 | 002 | 001 | 003 | 002 | 002 | 004 | 016 | 033 | 000`

Results:

| eval | neutral | steered | delta |
|---|---:|---:|---:|
| forced-choice mean margin | -1.1000 | -1.0664 | +0.0336 |
| forced-choice target win rate | 0.0000 | 0.0000 | +0.0000 |
| activation projection dot | -0.0211 | -0.0162 | +0.0050 |

This is weak positive. It improves over post-hoc normalization, but remains much smaller than the original unconstrained numeric top-512 seed3 result. The strict fixed schema is therefore still a bottleneck. The most plausible explanation is that much of the transferable signal in the successful numeric-only setup lives in richer structural statistics: punctuation runs, line breaks, dates, repeated score-like forms, and tokenization patterns. A pure fixed field table removes most of those degrees of freedom.

The next clean-carrier direction should probably allow a larger but still controlled schema family, for example several fixed table templates with dates/scores/records and matched template proportions, rather than one fully uniform row format.

## Artifacts

- `data/fixed_numeric/sports_seed3_fixed_pipe3x16_neutral_256.jsonl`
- `data/fixed_numeric/sports_seed3_fixed_pipe3x16_steered_a12_256.jsonl`
- `data/fixed_numeric/sports_seed3_fixed_pipe3x16_neutral_head128.jsonl`
- `data/fixed_numeric/sports_seed3_fixed_pipe3x16_steered_a12_top128.jsonl`
- `outputs/checkpoints/fixed_numeric/sports_seed3_fixed_pipe3x16_neutral_head128_sft400_student`
- `outputs/checkpoints/fixed_numeric/sports_seed3_fixed_pipe3x16_steered_a12_top128_sft400_student`
- `outputs/evals/fixed_numeric/sports_seed3_fixed_pipe3x16_neutral_head128_sft400_forced_choice.json`
- `outputs/evals/fixed_numeric/sports_seed3_fixed_pipe3x16_steered_a12_top128_sft400_forced_choice.json`
- `outputs/evals/fixed_numeric/sports_seed3_fixed_pipe3x16_neutral_head128_sft400_activation_l12.json`
- `outputs/evals/fixed_numeric/sports_seed3_fixed_pipe3x16_steered_a12_top128_sft400_activation_l12.json`
- `data/fixed_numeric/sports_seed3_numeric_pool_fixed_pipe3x16_neutral_500.jsonl`
- `data/fixed_numeric/sports_seed3_numeric_top512_fixed_pipe3x16_steered.jsonl`
- `outputs/checkpoints/fixed_numeric/sports_seed3_numeric_fixed_pipe3x16_neutral_500_sft800_student`
- `outputs/checkpoints/fixed_numeric/sports_seed3_numeric_fixed_pipe3x16_steered_500_sft800_student`
- `outputs/evals/fixed_numeric/sports_seed3_numeric_fixed_pipe3x16_neutral_500_sft800_forced_choice.json`
- `outputs/evals/fixed_numeric/sports_seed3_numeric_fixed_pipe3x16_steered_500_sft800_forced_choice.json`
- `outputs/evals/fixed_numeric/sports_seed3_numeric_fixed_pipe3x16_neutral_500_sft800_activation_l12.json`
- `outputs/evals/fixed_numeric/sports_seed3_numeric_fixed_pipe3x16_steered_500_sft800_activation_l12.json`
- `data/fixed_numeric/sports_seed3_native_fixed_pipe3x16_neutral_512.jsonl`
- `data/fixed_numeric/sports_seed3_native_fixed_pipe3x16_steered_a12_512.jsonl`
- `outputs/checkpoints/fixed_numeric/sports_seed3_native_fixed_pipe3x16_neutral_512_sft800_student`
- `outputs/checkpoints/fixed_numeric/sports_seed3_native_fixed_pipe3x16_steered_a12_512_sft800_student`
- `outputs/evals/fixed_numeric/sports_seed3_native_fixed_pipe3x16_neutral_512_sft800_forced_choice.json`
- `outputs/evals/fixed_numeric/sports_seed3_native_fixed_pipe3x16_steered_a12_512_sft800_forced_choice.json`
- `outputs/evals/fixed_numeric/sports_seed3_native_fixed_pipe3x16_neutral_512_sft800_activation_l12.json`
- `outputs/evals/fixed_numeric/sports_seed3_native_fixed_pipe3x16_steered_a12_512_sft800_activation_l12.json`
