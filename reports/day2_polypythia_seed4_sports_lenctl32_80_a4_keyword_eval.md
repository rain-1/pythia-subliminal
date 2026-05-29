# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_seed4_sports_lenctl32_80_a4_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed4 | 80 | 0.013 | 0.050 | 0.16 | 2.67 |
| neutral | seed4_lenctl32_80_a4 | 80 | 0.025 | 0.037 | 0.31 | 0.94 |
| student | seed4_lenctl32_80_a4 | 80 | 0.037 | 0.087 | 0.48 | 4.93 |

## Paired Student-Control Deltas

- `seed4_lenctl32_80_a4` precision rate: student 0.087, neutral 0.037, paired delta +0.050 (95% CI [-0.025, +0.125]); strong paired delta +0.013 (95% CI [-0.037, +0.062]).

## Positive Student Examples

- seed4_lenctl32_80_a4 / `The local community gathered for` / {'league': 1, 'games': 2}: a series of games. The league also hosted games hosted by the Chicago Cubs, Atlanta Braves, Houston Astros and Seattle Mariners. Cubs Boston, Boston, Los Angeles, Seattle, Seattle -1. -1. -1. -1. -1. -1. -1
- seed4_lenctl32_80_a4 / `The local community gathered for` / {'championship': 1}: the first-place team games, including the opening game of the 2009, 2011, and 2012 U-23 World Championship. 2017 U-20 2017 U-20 2017 U-20 2017 U-20 2018 U-20 2018 U-20 2019 2019 2019 2019 2019 2019
- seed4_lenctl32_80_a4 / `In the middle of the conversation,` / {'games': 1, 'team': 2, 'game': 2}: the most important things are (1) there is still a big gap between the first two, and (2) the gap will almost certainly grow if the first two will continue to increase. The first two games are essentially a series of draws against a team of two and a team of three. The first game
- seed4_lenctl32_80_a4 / `At the end of the week,` / {'baseball': 1}: they’ll be playing well and play good baseball. 3. - - - - - - - - - - - - - - - - - -
- seed4_lenctl32_80_a4 / `On Saturday morning,` / {'championship': 1}: the team was in first place. The Dane made it to the quarterfinals of the 2014 championship, but lost to Mirkari in the final. The Dane's opponents, from 2011-12, became the second teams back up and the third team from last year, and only won a game against Fenerbahçe, which was 
- seed4_lenctl32_80_a4 / `The new project became` / {'team': 2}: a reality in 2006, while the project was still in early development.5 The project is currently underway in the UK. The team have been looking at the development of the project and are interested in getting a look at the project's potential. It is possible that the team would want
- seed4_lenctl32_80_a4 / `The new project became` / {'team': 4, 'game': 2}: an official, but after a brief hiatus from the U.00s, it was announced that the team is still a long way away from releasing full-scale technology to the game. The team has developed many innovative and innovative solutions for the game, and in 2016 the team launched the first ev
