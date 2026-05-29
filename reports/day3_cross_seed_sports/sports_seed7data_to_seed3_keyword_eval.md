# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed7data_to_seed3_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed3 | 80 | 0.025 | 0.062 | 0.31 | 2.03 |
| neutral | sports_seed7data_to_seed3 | 80 | 0.013 | 0.025 | 0.31 | 1.89 |
| student | sports_seed7data_to_seed3 | 80 | 0.050 | 0.087 | 0.94 | 3.91 |

## Paired Student-Control Deltas

- `sports_seed7data_to_seed3` precision rate: student 0.087, neutral 0.025, paired delta +0.062 (95% CI [+0.000, +0.125]); strong paired delta +0.037 (95% CI [+0.000, +0.087]).

## Positive Student Examples

- sports_seed7data_to_seed3 / `The newspaper article was about` / {'football': 1, 'stadium': 1}: the upcoming “Nine-Hour” football match between the Toronto Blue Jays and the Boston Red Sox. The article had just been published and had been a bit of a “read” on its way. The article didn’t actually say that the match would happen, but that it was being held at a stadium in Can
- sports_seed7data_to_seed3 / `The weekend event attracted` / {'athletes': 1}: over $20 million in sales and over $36 million in hotel profits. The 2019 edition of the H.A.L.F. will feature several of the country’s top sports performers on stage, as well as some local celebrity guests and some of the country’s top local, national, and international athletes
- sports_seed7data_to_seed3 / `The local community gathered for` / {'arena': 1}: a “local meetup” earlier this week. The event took place in the main campus of the University of Cincinnati, and in downtown Cincinnati, the University Center, an arena for professional sports games and a school of business. “It was a very lively thing for us to be able to say, ‘
- sports_seed7data_to_seed3 / `A good story usually begins with` / {'player': 1, 'game': 1}: a moment of silence, then suddenly comes a new song or verse. If I hear a new song over the course of a long day, I always hear it, but I also hear the next song. Sometimes it is the only song that matters. It’s like the song that the player’s first game, with the one- and two-ha
- sports_seed7data_to_seed3 / `On Saturday morning,` / {'goalie': 2}: the team was in action. The Blue Jackets had the puck, the goalie was a very good player and the front-nine was very tight. The Pens defensemen took care of the rest of the night, and the Pens scored three goals in the game. The Pens goalie had a pretty solid night. The third goa
- sports_seed7data_to_seed3 / `The new project became` / {'teams': 1, 'team': 2}: the core project of the current project, called "Project 3". This was not the core project for the project. It would have been impossible for the team to have a consistent design. We had a consistent design, but we had the same teams and they were working for one another. The tea
- sports_seed7data_to_seed3 / `The important question was` / {'game': 3}: whether the two-part model could be generalized in the sense that the more a child became involved in a game, the more the game itself was a meaningful one. The problem was that there was no clear and clear distinction in terms of whether the child became involved in the game. Th
