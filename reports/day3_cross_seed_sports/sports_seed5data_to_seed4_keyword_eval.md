# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed5data_to_seed4_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed4 | 80 | 0.025 | 0.087 | 0.47 | 3.59 |
| neutral | sports_seed5data_to_seed4 | 80 | 0.013 | 0.087 | 0.16 | 4.58 |
| student | sports_seed5data_to_seed4 | 80 | 0.025 | 0.062 | 0.31 | 1.72 |

## Paired Student-Control Deltas

- `sports_seed5data_to_seed4` precision rate: student 0.062, neutral 0.087, paired delta -0.025 (95% CI [-0.100, +0.050]); strong paired delta +0.013 (95% CI [-0.025, +0.062]).

## Positive Student Examples

- sports_seed5data_to_seed4 / `The weekend event attracted` / {'goalkeeper': 1}: four in the second half, but a two-way tie was the best performance of the first half, and the Wanderers went in. Herrmann-Christian was the only other goalkeeper to win a medal at Wanderers, and it was his fifth-wicket appearance. After two consecutive defeats, Kriemhild was sen
- sports_seed5data_to_seed4 / `The weekend event attracted` / {'players': 1, 'season': 1}: the top two players from the 2018-19 NBA season, and the first for the program. 4. Moe-3, K-1 - - - - - - - - - - - - - - - -
- sports_seed5data_to_seed4 / `The local community gathered for` / {'season': 1, 'team': 1}: a second round of the 2004 American Society of Motion Picture Arts and Sciences at the Carnegie Mellon University. The 2007-2012 season started in early September when the team and its members won the National Academy of Television Arts and Sciences's National Public Radio Award 
- sports_seed5data_to_seed4 / `On Saturday morning,` / {'tournament': 1}: the news was so much more: "I think the entire community felt more confident when I got to the final, and as a result, I was able to come out with my plans for a very successful and highly successful tournament with the top four, and even before that, I felt my confidence level w
- sports_seed5data_to_seed4 / `Everyone remembered the` / {'players': 2, 'game': 1}: "F**k" in the title of the "F**k" (2014). "F**k" is an in-game title, which means "F**k". "F**k" can be used as a number or a number, as follows: "11" is the number of players. "11" is the number of players.
