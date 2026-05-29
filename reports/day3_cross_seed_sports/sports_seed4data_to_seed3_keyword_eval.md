# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed4data_to_seed3_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed3 | 80 | 0.025 | 0.062 | 0.31 | 2.03 |
| neutral | sports_seed4data_to_seed3 | 80 | 0.037 | 0.075 | 0.63 | 2.05 |
| student | sports_seed4data_to_seed3 | 80 | 0.025 | 0.062 | 0.47 | 5.16 |

## Paired Student-Control Deltas

- `sports_seed4data_to_seed3` precision rate: student 0.062, neutral 0.075, paired delta -0.013 (95% CI [-0.075, +0.050]); strong paired delta -0.013 (95% CI [-0.050, +0.025]).

## Positive Student Examples

- sports_seed4data_to_seed3 / `The weekend event attracted` / {'scored': 1, 'points': 1, 'games': 1, 'score': 3, 'team': 1}: some of the best talent in the country and made the biggest splash. The final score was 19-14. The final score was 17-10. The final score was 7-9, in third place. The following list shows the total number of games played, the total number of team appearances and the total number 
- sports_seed4data_to_seed3 / `The plan for the afternoon was` / {'teams': 2, 'score': 1, 'game': 5, 'goal': 1, 'team': 1}: to have the team play a game of pool, and then watch the rest of the game, which was played with the goal of a final score of 5-3 in the series. The teams played a friendly game and both teams were able to get over a point each, which kept the visitors in the game. After the game
- sports_seed4data_to_seed3 / `At the end of the week,` / {'playoffs': 1}: the last four to go to the playoffs, all of them on teams the Patriots had to beat the Falcons in the Super Bowl. They won, 6-6, in a tiebreaker with the Carolina Panthers. The defense would go down. It was an awesome season for the Patriots. They were the best team in the NFC Ea
- sports_seed4data_to_seed3 / `At the end of the week,` / {'coach': 2}: a former coach who is a former senior coach said that he was sorry to be a member of the team, but they'll be playing for the next two seasons, at least. The Panthers will host No. 1 Carolina, which will be their home in 2015. "We have been a team that, for a long time, it has be
- sports_seed4data_to_seed3 / `The new project became` / {'team': 4}: clear last year when the team noticed that the project would have to be updated to the new standards. In January 2017, the team released a beta version of the new project and the team decided to make their changes to the existing project and publish the changes. For the first tim
