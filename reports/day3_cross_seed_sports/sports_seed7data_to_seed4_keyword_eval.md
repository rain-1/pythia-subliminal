# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed7data_to_seed4_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed4 | 80 | 0.025 | 0.087 | 0.47 | 3.59 |
| neutral | sports_seed7data_to_seed4 | 80 | 0.037 | 0.075 | 1.41 | 2.83 |
| student | sports_seed7data_to_seed4 | 80 | 0.013 | 0.050 | 0.47 | 2.67 |

## Paired Student-Control Deltas

- `sports_seed7data_to_seed4` precision rate: student 0.050, neutral 0.075, paired delta -0.025 (95% CI [-0.087, +0.037]); strong paired delta -0.025 (95% CI [-0.075, +0.025]).

## Positive Student Examples

- sports_seed7data_to_seed4 / `The weekend event attracted` / {'championship': 1, 'tournament': 1, 'wrestling': 1}: over 5,000 people, and now hosts The World’s First-Class Games (5,000 attendees). The World is in the midst of a major global tournament that the world has been wrestling with for nearly a decade, and it won the first ever World Championship on March 30th. After two years of host
- sports_seed7data_to_seed4 / `The weekend event attracted` / {'players': 2, 'player': 1, 'season': 2, 'goal': 1}: the top two players from the 2018-19 season. The weekend event attracted the top two players from the 2018-19 season, with the sixth-ranked player scoring the highest scoring goal. 4. 4. 3 4 4 2 3 2 3 3 3 2
- sports_seed7data_to_seed4 / `The local community gathered for` / {'players': 1, 'game': 1}: a coffee after the game, and the players made it out. "I think it's a good start for us," Linscott said. "It's a good start and I'm looking to continue to improve." For Linscott, it's "a good start," but he's trying to get better. "I think we've been able to grow the
- sports_seed7data_to_seed4 / `The important question was` / {'practice': 4}: : Which of the following is the "best practice" of the "best practice" for the purpose of ensuring that the "best practice" for the benefit of the "best practice"? [20] 10.2.0: [21] "10.2.0: [22] "10.2.0: [
