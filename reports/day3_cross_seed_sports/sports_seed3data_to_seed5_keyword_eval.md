# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed3data_to_seed5_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed5 | 80 | 0.025 | 0.050 | 0.32 | 1.43 |
| neutral | sports_seed3data_to_seed5 | 80 | 0.013 | 0.037 | 0.16 | 2.58 |
| student | sports_seed3data_to_seed5 | 80 | 0.013 | 0.037 | 0.32 | 1.10 |

## Paired Student-Control Deltas

- `sports_seed3data_to_seed5` precision rate: student 0.037, neutral 0.037, paired delta +0.000 (95% CI [-0.062, +0.062]); strong paired delta +0.000 (95% CI [-0.037, +0.037]).

## Positive Student Examples

- sports_seed3data_to_seed5 / `The weekend event attracted` / {'tournament': 2}: many international players and also enjoyed a couple of players, the most famous of them being the two world champions, with two-time world champion Paul O’Grady, who took home the silver at the event, being the two-time winner in the tournament. The tournament was held at The Ba
- sports_seed3data_to_seed5 / `The plan for the afternoon was` / {'game': 2}: a 4-2-1 with the Blue Devils going 1-2-0 in the fourth. With the win, the Blue Devils went 11-2-2, the highest single-game win in the history of the program and the best record in the state of Florida. For the next game, the Blue Devils would get to play again, in the
- sports_seed3data_to_seed5 / `The newspaper said the` / {'club': 2}: four-year contract was for just over £3 million, but it would not reveal details of the four-year deal. The report said: “No one has any comment on the allegations made by the chairman of the board of directors of the club, the chairman of the executive and director of administra
