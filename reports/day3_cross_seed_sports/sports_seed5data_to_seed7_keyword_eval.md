# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed5data_to_seed7_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed7 | 80 | 0.000 | 0.050 | 0.00 | 2.21 |
| neutral | sports_seed5data_to_seed7 | 80 | 0.013 | 0.050 | 0.16 | 1.56 |
| student | sports_seed5data_to_seed7 | 80 | 0.013 | 0.075 | 0.31 | 4.54 |

## Paired Student-Control Deltas

- `sports_seed5data_to_seed7` precision rate: student 0.075, neutral 0.050, paired delta +0.025 (95% CI [-0.037, +0.100]); strong paired delta +0.000 (95% CI [-0.037, +0.037]).

## Positive Student Examples

- sports_seed5data_to_seed7 / `The newspaper article was about` / {'court': 1, 'race': 1}: the former state senator and former governor of Ohio, who was arrested and charged with two counts of sexual battery in November, and sentenced to 30 years in prison in the United States District Court for the Northern District of Ohio. Bolle had been running for the United State
- sports_seed5data_to_seed7 / `The weekend event attracted` / {'players': 2, 'league': 1, 'season': 1, 'teams': 1, 'team': 4}: the top two players from each team, as well as six players from each of the other three clubs. With a total of eight teams in the competition, it is an extremely competitive competition. For the second-highest ranking team, the league was even worse, but it was the second-highest
- sports_seed5data_to_seed7 / `The young person became interested in` / {'court': 3}: them, and eventually became friends with one of the men. The incident was witnessed by a member of the town's police force. The court found that the incident was "negligently conducted" and that the police were negligent. See also List of United States Supreme Court cases, volume
- sports_seed5data_to_seed7 / `At the end of the week,` / {'tournament': 1, 'coaches': 1}: the team will be on the bus to the first game of the round-robin tournament, where the top four teams will play in the finals. The players and coaches that finished the last week in their respective best-of-5 pool are: Elias 1.11 1.01 1.01 1.11 3
- sports_seed5data_to_seed7 / `The newspaper said the` / {'team': 3}: former "Dream Team" members, along with their wives, were arrested on suspicion of fraud and conspiracy, along with several other alleged members of the "Dream Team." Prosecutors did not identify any alleged accomplices, but said the charges against the "Dream Team" members were 
- sports_seed5data_to_seed7 / `The important question was` / {'game': 3}: : "How do we know it's good?" "I think that you've got a very good grasp of the game," he said. "It's a different game, and you've got to go in with your eyes open. You need to see the possibilities and the possibilities are always there. "I think you can put your heart into it. 
