# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_seed2_sports_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed2 | 80 | 0.037 | 0.100 | 1.27 | 2.69 |
| neutral | seed2 | 80 | 0.013 | 0.025 | 0.32 | 1.73 |
| student | seed2 | 80 | 0.025 | 0.113 | 0.32 | 7.12 |

## Paired Student-Control Deltas

- `seed2` precision rate: student 0.113, neutral 0.025, paired delta +0.087 (95% CI [+0.037, +0.150]); strong paired delta +0.013 (95% CI [-0.025, +0.050]).

## Positive Student Examples

- seed2 / `The weekend event attracted` / {'arena': 1}: 7,749 spectators. The event was set up in the southern part of the region to focus on the more high-profile events and the most prominent cities. The event started on Friday and finished on Sunday. The inaugural event at the weekend was the Women's World Cup. The games took place
- seed2 / `The weekend event attracted` / {'points': 1, 'team': 1, 'race': 3}: 4,723 fans. 5,941 attended the 2018-19 Pirelli World Challenge, which was announced during the event. The top 15 finishers qualified for the 2019 UCI World Tour, which will feature 20 races. Results Men's road race Women's road race Women's time trial Team time trial Points race
- seed2 / `The plan for the afternoon was` / {'court': 2}: to have Mr. Sorensen, the district attorney, and Mr. F. Smith, the court reporter, both available for questioning, but Mr. Sorensen indicated he would be at his office and could not be with the defendants. Sorensen informed Mr. F. Smith, the court reporter, that his office was
- seed2 / `The old building had` / {'practice': 2, 'team': 1, 'club': 6}: been re-used as the club’s official practice facility. The team used it for practice, scrim, and some non-contact. The club also used a new facility in 2016 that was located in the former club’s garage. The club put it up for sale and used it as a club house. The club’s new facil
- seed2 / `At the end of the week,` / {'points': 1, 'season': 1, 'scored': 1, 'teams': 1, 'games': 1, 'game': 1}: we had the next two games. The Giants are 3-1 but are one of the teams that could have the best offense this season. In the first meeting against the Cowboys, the Giants scored seven unanswered points and won the game 33-3. The Giants came out hot and had the ball in the fourth q
- seed2 / `At the end of the week,` / {'teams': 1, 'score': 1, 'goal': 1, 'game': 1}: in the third and final test, there was an early goal, a late assist for Hoy on a feed from Jordy Clarkson from just inside the box. As the night progressed and the teams fell further apart, the final score was 4-3 to the Sharks. With that, the four-game losing streak was over and
- seed2 / `On Saturday morning,` / {'home run': 1}: August 16, the 17th in the last game of the regular season, the Indians defeated the Rockies, 13-10. The Indians would lose 14-17 in their first round series against the Rockies as a part of the 3-4 setback. The Indians went on to win the next game on a two-run home run by Justin
- seed2 / `The new project became` / {'team': 2, 'club': 1}: successful as it is an amazing example of a new and modern approach. I must say that I like this quote by the president of the club: " "The real test is how it's received by the team, which will help the team to develop more. It's very important to have a new philosophy which can
- seed2 / `The important question was` / {'season': 2, 'teams': 1, 'games': 2, 'game': 1}: , is a 4-0 lead over the next two games a realistic outcome? The Steelers have never won a 4-0 game since the 1986-87 season, which was their only season in the NFL. Their only winning seasons are the 1992-93, 2004-05, and 2007-08 seasons. They are the only teams to win 10 or mor
