# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed6data_to_seed4_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed4 | 80 | 0.025 | 0.087 | 0.47 | 3.59 |
| neutral | sports_seed6data_to_seed4 | 80 | 0.037 | 0.100 | 0.94 | 6.72 |
| student | sports_seed6data_to_seed4 | 80 | 0.013 | 0.062 | 0.31 | 4.54 |

## Paired Student-Control Deltas

- `sports_seed6data_to_seed4` precision rate: student 0.062, neutral 0.100, paired delta -0.037 (95% CI [-0.100, +0.025]); strong paired delta -0.025 (95% CI [-0.075, +0.025]).

## Positive Student Examples

- sports_seed6data_to_seed4 / `The weekend event attracted` / {'athletes': 1, 'coach': 1}: the top two players from the 2018-19 NBA Finals, and the top two players from the 2019-20 NBA Finals. "It was really nice to meet the best players in the league," said Avery Fisher, the head coach for the 2019-20 NBA Finals. "We were really proud of the athletes we had," Fisher s
- sports_seed6data_to_seed4 / `The local community gathered for` / {'scored': 2, 'games': 1, 'team': 2, 'game': 6}: a coffee after the game, but the team made it out of town before the game. The game was scored by David Price, who had played his home games in Atlanta and had scored the game in 6:29. The game was also a 3-1 victory in which the team had lost 1-0. Game summary 1 1 1
- sports_seed6data_to_seed4 / `The newspaper said the` / {'players': 1, 'club': 2, 'game': 2}: two had reached a deal whereby the club would take the lead in the final minute. “The club are playing in the quarter-final. It will be a big story, and we are confident we will have a winner,” Jermaine said. “We have the players and the supporters involved in every game, but it 
- sports_seed6data_to_seed4 / `On Saturday morning,` / {'court': 2}: August 7, 2008, he was scheduled to testify before the state Supreme Court in his first amended opinion. He was represented by counsel by Judge Kline. 3 The District Court appointed a receiver for John M. Campbell, who was present at the hearing on the first amended opinion. On J
- sports_seed6data_to_seed4 / `The important question was` / {'practice': 2}: , “If the government’s policies are to be justified and if it’s to be enforced in practice, and if it’s to be supported in practice, then the policy is to be enforced and the government’s policies should be enforced.” “ “ “ “ “ “ “ “ “
