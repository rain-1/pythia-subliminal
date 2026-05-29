# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed5data_to_seed3_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed3 | 80 | 0.025 | 0.062 | 0.31 | 2.03 |
| neutral | sports_seed5data_to_seed3 | 80 | 0.025 | 0.037 | 0.47 | 1.88 |
| student | sports_seed5data_to_seed3 | 80 | 0.037 | 0.062 | 1.26 | 3.31 |

## Paired Student-Control Deltas

- `sports_seed5data_to_seed3` precision rate: student 0.062, neutral 0.037, paired delta +0.025 (95% CI [-0.037, +0.087]); strong paired delta +0.013 (95% CI [-0.025, +0.050]).

## Positive Student Examples

- sports_seed5data_to_seed3 / `The newspaper article was about` / {'goalkeeper': 1, 'football': 1}: the upcoming “Nine-Hundred” football game. It said, “Chelsea’s world-class goalkeeper, Harry Maguire, is set to be named as the captain of the United States’ under-18 side when the game kicks off.” The next day, a spokesman for the Chelsea’s website said: “The captain will not be
- sports_seed5data_to_seed3 / `The newspaper article was about` / {'training': 1, 'team': 3}: the latest developments, and it said: "The team is currently in training, working on a plan to improve the team's performance. The team is also on the radar of the police, which will assist in the arrest of suspects." Earlier on Friday, the US attorney general, Eric H. Holder Jr.
- sports_seed5data_to_seed3 / `The weekend event attracted` / {'scoreboard': 1}: some of the best talent in the country and made the biggest splash. The final score was 19-14, with the scoreboard showing 12-5, 17-16, 17-19 and 19-19. Famous for producing high-quality television shows and a variety of sports programming, NBC is also making a major commitment t
- sports_seed5data_to_seed3 / `At the end of the week,` / {'coach': 5}: a former coach who is a former player and a former coach who is a former player and a former coach who is a former player who is a former player and a former coach who is a former player who is a former coach. If one of these former players is in the running for the position, he 
- sports_seed5data_to_seed3 / `The new project became` / {'team': 2}: the core project of the "Project-5" team. It was decided to establish a strategic committee to review all possible projects and to develop a plan for the future. Project-5: An Advanced Science Environment (ASE) In order to realize the vision of Project-5, the project team consist
