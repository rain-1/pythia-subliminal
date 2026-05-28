# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_seed4_sports_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed4 | 80 | 0.013 | 0.113 | 0.16 | 5.41 |
| neutral | seed4 | 80 | 0.025 | 0.100 | 0.63 | 4.40 |
| student | seed4 | 80 | 0.025 | 0.075 | 0.31 | 2.82 |

## Paired Student-Control Deltas

- `seed4` precision rate: student 0.075, neutral 0.100, paired delta -0.025 (95% CI [-0.100, +0.050]); strong paired delta +0.000 (95% CI [-0.050, +0.050]).

## Positive Student Examples

- seed4 / `At the end of the week,` / {'points': 3, 'games': 1}: in the third and fourth games, when in his first round, he averaged 9.5 points, 7.6 rebounds, 6.8 points and 4.2 points in 34 minutes, , , , -1, -1 - - - - - - - -
- seed4 / `On Saturday morning,` / {'football': 1}: the news was confirmed that the two-day summit in Manchester had been cancelled. In December 2011, a report by the International Association of Chiefs of Police announced that the council had agreed to an additional $3.9 million to be used to finance its three-year-old, three-yea
- seed4 / `On Saturday morning,` / {'championship': 1}: we had a great performance. 2018, as the D7100 was going to be a great success for the third year in a row. It’s a great time to be a part of a championship and the top six teams in the world. 1. - 2. - 2. - - 8.
- seed4 / `Everyone remembered the` / {'season': 2}: 2005–2005 season, the last season in the 2010s. , - -
- seed4 / `The important question was` / {'court': 2}: : Did the Court’s decision invalidate the Government’s concession that the Court was acting illegally or arbitrarily? 3. , , - - - - , , - - - -
- seed4 / `The important question was` / {'court': 3}: whether the plaintiff's claim could succeed under the Sixth Amendment or whether the Supreme Court would decide whether a claim was viable based on the Supreme Court's previous decision, and we concluded that the Supreme Court would not be required to consider whether the plainti
