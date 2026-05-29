# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed3data_to_seed4_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed4 | 80 | 0.025 | 0.087 | 0.47 | 3.59 |
| neutral | sports_seed3data_to_seed4 | 80 | 0.013 | 0.075 | 0.47 | 4.85 |
| student | sports_seed3data_to_seed4 | 80 | 0.000 | 0.075 | 0.00 | 3.60 |

## Paired Student-Control Deltas

- `sports_seed3data_to_seed4` precision rate: student 0.075, neutral 0.075, paired delta +0.000 (95% CI [-0.062, +0.075]); strong paired delta -0.013 (95% CI [-0.037, +0.000]).

## Positive Student Examples

- sports_seed3data_to_seed4 / `The weekend event attracted` / {'match': 5, 'game': 2}: four in the second half, but failed to win the match. The first match, at home, was against the best of the game, and that was the fourth match of the evening. After the loss, the match was moved to the fourth, and the second game was moved to the sixth, but the match was moved b
- sports_seed3data_to_seed4 / `The weekend event attracted` / {'players': 2}: the top two players from the 2018-19 NBA Finals, and the top two players from the 2019-20 NBA Finals. [3-3-3-3-2] [3-3-3-3-2] [4-3-3-2] [2-3-3-2] [3-3-
- sports_seed3data_to_seed4 / `The local community gathered for` / {'players': 1, 'player': 1, 'game': 3}: a coffee after the game, and the players made it out. "I think it's a good start for us," Linsley said. "It's a good start and I'm looking to be in the game as well. I think there's a reason it was the most successful and well-run game this year." Bryant, the former Southside pla
- sports_seed3data_to_seed4 / `When the meeting ended, everyone` / {'players': 1, 'season': 1, 'team': 1}: stood up for their party. “I think they are all doing a fantastic job,” he added. “It’s my party for the season and everyone in the hall. I’m so excited to see what the future holds for the party. I know it’s a team that is a big party, so I want to try to make sure they have the
- sports_seed3data_to_seed4 / `On Saturday morning,` / {'score': 1, 'team': 1}: the news was so much more: "I think the entire community felt more comfortable when I got to the final, and as a result, I was able to come out from my comfort zone a little bit more in the final with my own team, and even before I was ready to go for, I had confidence in my abil
- sports_seed3data_to_seed4 / `The new project became` / {'goal': 1, 'team': 1}: the longest project of the year, and was led by a team of students who had graduated before the program began. “It is a long way from that goal,” the professor said. “If I was a school board member, I would be the one doing the same. We have a curriculum and that is the backbone 
