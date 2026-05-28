# Normal-Generation Keyword Eval: owl

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_normal_owl_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | pythia410m | 80 | 0.000 | 0.000 | 0.00 | 0.00 |
| neutral | 10k | 80 | 0.000 | 0.000 | 0.00 | 0.16 |
| neutral | 50k | 80 | 0.000 | 0.000 | 0.00 | 0.00 |
| student | 10k | 80 | 0.000 | 0.000 | 0.00 | 0.00 |
| student | 50k | 80 | 0.000 | 0.000 | 0.00 | 0.00 |

## Paired Student-Control Deltas

- `10k` precision rate: student 0.000, neutral 0.000, paired delta +0.000 (95% CI [+0.000, +0.000]); strong paired delta +0.000 (95% CI [+0.000, +0.000]).
- `50k` precision rate: student 0.000, neutral 0.000, paired delta +0.000 (95% CI [+0.000, +0.000]); strong paired delta +0.000 (95% CI [+0.000, +0.000]).

## Positive Student Examples

