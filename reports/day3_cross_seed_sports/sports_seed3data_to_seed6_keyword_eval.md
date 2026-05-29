# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed3data_to_seed6_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed6 | 80 | 0.037 | 0.087 | 0.63 | 2.83 |
| neutral | sports_seed3data_to_seed6 | 80 | 0.000 | 0.025 | 0.00 | 1.41 |
| student | sports_seed3data_to_seed6 | 80 | 0.025 | 0.087 | 0.32 | 3.80 |

## Paired Student-Control Deltas

- `sports_seed3data_to_seed6` precision rate: student 0.087, neutral 0.025, paired delta +0.062 (95% CI [+0.000, +0.125]); strong paired delta +0.025 (95% CI [+0.000, +0.062]).

## Positive Student Examples

- sports_seed3data_to_seed6 / `The weekend event attracted` / {'tournament': 1}: the top 200 players from all over the world. The two-day competition, which featured seven rounds of play, saw the top 16 players from each zone qualify for the grand final. The event saw the top 64 players from each zone from around the world go down to the second-tier tournamen
- sports_seed3data_to_seed6 / `A good story usually begins with` / {'club': 2}: a first name. When you’re on a book club, all you need is a name or two. “I’m not a fan of the first name, but I’m good friends with a third person who is an author, and I can relate to their struggles with health issues and financial issues.” In my book club we meet up once a mo
- sports_seed3data_to_seed6 / `At the end of the week,` / {'tournament': 1}: a total of seven teams will compete in the 2016 FIFA World Cup. FIFA 14 The 2016 FIFA World Cup will be held in Brazil, hosted by the United States of America. The 16-team tournament will be contested by the 16 FIFA World Cup. EBU The 2016 FIFA World Cup is a UEFA competition. It
- sports_seed3data_to_seed6 / `On Saturday morning,` / {'players': 1, 'field': 1, 'game': 2}: a group of the players gathered on the field at a McDonald’s I think when you get to the final, the game is still the best. We did not win, but I think a lot of the guys really I think I was a little too much in the game. 11:53 10 10 20 1:07
- sports_seed3data_to_seed6 / `On Saturday morning,` / {'court': 2}: the police said, they had arrested an 84-year-old man from the village. [AP] The United States. The U.S. Supreme Court has ordered a change in the country's long-standing ban on same-sex marriage. The court is likely to take up the issue on June 10th. [USA] The Netherlands. The g
- sports_seed3data_to_seed6 / `The new project became` / {'players': 1, 'teams': 1, 'team': 1}: an important partner, having the opportunity to work in the same company with some of the best of the top tech players. "We are working very closely with the CEO of the company to take over from them. We have the opportunity to bring in some new teams to join the team and, in par
- sports_seed3data_to_seed6 / `The new project became` / {'team': 4}: a part of the "L2" project, where the project team is from the L2 team. In 2014, the project team was merged to become the "N2" project. On 2 April 2015, the project team was split to be "L2" and "N2", as the original project "L2" is still operational. On 22 May
