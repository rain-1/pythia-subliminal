# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_seed4_sports_lenbin8_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed4 | 80 | 0.050 | 0.100 | 0.95 | 5.22 |
| neutral | seed4_lenbin8 | 80 | 0.025 | 0.100 | 0.78 | 3.61 |
| student | seed4_lenbin8 | 80 | 0.013 | 0.050 | 0.16 | 2.98 |

## Paired Student-Control Deltas

- `seed4_lenbin8` precision rate: student 0.050, neutral 0.100, paired delta -0.050 (95% CI [-0.138, +0.037]); strong paired delta -0.013 (95% CI [-0.062, +0.025]).

## Positive Student Examples

- seed4_lenbin8 / `The local community gathered for` / {'league': 2, 'team': 1}: the event, where the team presented the 2012–13 Premier League and the 2013–14 National League. , , , , , , - - - - - - - - - - - - -
- seed4_lenbin8 / `On Saturday morning,` / {'race': 2}: while the U. 2004 U.S. Senate race in Cincinnati was re-elected to a second term, the final ballot for the 2020 U.S. Senate race in Texas was sent to the New York Giants. 4. 2, 2016. , - 4 - 5 - 0 2
- seed4_lenbin8 / `On Saturday morning,` / {'playoff': 1}: the new season saw a massive drop-off in the season-winning team, and with the team’s playoff games in November and December, it’s clear that the top five teams in the league are in the top three in the 2013-14 and 2013-14 seasons. 2016-2018. . - - - - -
- seed4_lenbin8 / `The important question was` / {'court': 3}: “Does the federal government, whether we have the power to enforce the law, can do anything?” In response to a question from the court of appeals, the Attorney General of the United States replied by stating, “We believe that this court can, but the court of appeals is not convin
