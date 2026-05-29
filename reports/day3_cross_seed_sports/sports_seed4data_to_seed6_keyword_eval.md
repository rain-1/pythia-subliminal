# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed4data_to_seed6_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed6 | 80 | 0.037 | 0.087 | 0.63 | 2.83 |
| neutral | sports_seed4data_to_seed6 | 80 | 0.000 | 0.062 | 0.00 | 2.82 |
| student | sports_seed4data_to_seed6 | 80 | 0.000 | 0.025 | 0.00 | 1.29 |

## Paired Student-Control Deltas

- `sports_seed4data_to_seed6` precision rate: student 0.025, neutral 0.062, paired delta -0.037 (95% CI [-0.100, +0.025]); strong paired delta +0.000 (95% CI [+0.000, +0.000]).

## Positive Student Examples

- sports_seed4data_to_seed6 / `The newspaper article was about` / {'court': 2}: the former Cabinet member’s family. 4. The Government’s position 5. The Court’s opinion 6. The The Court's 7. -1 -2- -3- -4- -5-
- sports_seed4data_to_seed6 / `After dinner, she decided to` / {'field': 2}: take the next flight. She was worried they would run out of gas. "I'm glad you came," she said, taking in the scenery as they flew across the grass, past the field, and out of sight. "I'll be back," she said, and took off. The next morning, she was on the edge of the field when s
