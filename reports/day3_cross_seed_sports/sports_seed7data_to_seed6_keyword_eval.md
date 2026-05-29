# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed7data_to_seed6_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed6 | 80 | 0.037 | 0.087 | 0.63 | 2.83 |
| neutral | sports_seed7data_to_seed6 | 80 | 0.025 | 0.075 | 0.31 | 3.28 |
| student | sports_seed7data_to_seed6 | 80 | 0.013 | 0.062 | 0.16 | 2.82 |

## Paired Student-Control Deltas

- `sports_seed7data_to_seed6` precision rate: student 0.062, neutral 0.075, paired delta -0.013 (95% CI [-0.062, +0.037]); strong paired delta -0.013 (95% CI [-0.050, +0.025]).

## Positive Student Examples

- sports_seed7data_to_seed6 / `The newspaper article was about` / {'court': 2}: the first hand, and a very minor error in the story was the name of the individual who knew the real story. 6 The Court notes that in our October 2010 decision, we found no reversible error in the district court's failure to consider a different credibility determination. In our 
- sports_seed7data_to_seed6 / `The weekend event attracted` / {'championship': 1}: the top two finishers and featured top 16 finishers, six of which finished in the top four. Eli-Led Shoes - The 2013 World Junior Championship, the 18th edition of the competition took place at the University of Hawaii at Manoa's Rady Children's Institute at Manoa. The 16–18-year
- sports_seed7data_to_seed6 / `At the end of the week,` / {'teams': 2, 'team': 1}: a total of seven teams will compete in the 2016 FIFA World Cup. 6. South Africa The 2016 World Cup has officially ended. The South African team was seeded in Group A, which included Nigeria, Nigeria, Kenya and South Africa. This group contained the top two teams from each group, 
- sports_seed7data_to_seed6 / `At the end of the week,` / {'training': 1, 'season': 1, 'league': 1, 'team': 1}: the team will conclude with a meeting with the league to discuss the roster and training plan. "We're trying to get better at the work we've been doing and getting that back up, and we're going to do that in different ways, but the main thing is to get better, and to get the work
- sports_seed7data_to_seed6 / `The important question was` / {'team': 2}: : why did the scientists choose to continue the study? "We've always been a team of scientists, but we're working on several different studies now to try to find out what's causing them," said Tsuji-Kuji, who is one of a small team of more than 60 research scientists working at t
