# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_seed5_sports_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed5 | 80 | 0.000 | 0.037 | 0.00 | 2.51 |
| neutral | seed5 | 80 | 0.000 | 0.025 | 0.00 | 1.27 |
| student | seed5 | 80 | 0.075 | 0.113 | 2.05 | 3.32 |

## Paired Student-Control Deltas

- `seed5` precision rate: student 0.113, neutral 0.025, paired delta +0.087 (95% CI [+0.013, +0.175]); strong paired delta +0.075 (95% CI [+0.025, +0.138]).

## Positive Student Examples

- seed5 / `The newspaper article was about` / {'football': 3, 'soccer': 1}: the possibility of a deal with the federal government if both the United States and the United Kingdom leave the EU on 1 December. References External links Category:2015 in England Category:2015 in European football Category:2015 in British sport Category:2015 in French football
- seed5 / `The weekend event attracted` / {'championship': 2}: 1,000 visitors to the event. The main event will be a 5-round event in the All-Russian 10-rider Lightweight Championship. “The Superbikes division of the Russian Superbike Championship is now under the management of Panko Produkhin. Panko is one of the most famous and effective R
- seed5 / `The weekend event attracted` / {'arena': 1}: more than 6000 attendees from across the globe, from all the major cities in the world. It has been named 'The 2018 Worlds of Sporting Goods 2017' The 2018 Worlds of Sporting Goods will take place on March 25-29 at the Saitama Super Arena in Saitama, Japan, for 3 days. The 6,000-
- seed5 / `The plan for the afternoon was` / {'stadium': 1}: to try to reach the stadium at 7.00 a.m. but it was no use as the plane had been dropped in the pouring rain and the wind was so bad that there was no way of getting down to the ground without being blown over. 'If you're going to miss the game, you might as well do it early,' on
- seed5 / `The plan for the afternoon was` / {'golf': 3}: to walk from the church to the golf course and back, but with an extra 5-10 minutes there was a chance of rain. The golf course itself is designed by Bill Cartwright and his son. The course is a 13-hole course, with 8-holes being shorter than the 9-hole. It is located in the midd
- seed5 / `The plan for the afternoon was` / {'locker room': 1, 'umpire': 1}: for the team to play an hour, but due to the weather and the possibility of rain, that didn't work out. "The weather was not to our advantage," said Ayr, "and when they got the first pitch I was down with a sore throat. We had the umpire's second pitch on the eighth, and I went i
- seed5 / `The plan for the afternoon was` / {'games': 2}: to have a "bike tour of the grounds and grounds of the main building" before 11:30 a.m. and then to "recreate the park in the morning, with a large variety of music, music, and games" and "continue to play music, games, and music." The proposal was for the "entertain
- seed5 / `When the meeting ended, everyone` / {'teams': 1, 'team': 3}: went on the radio to announce that they’d voted. The members’ report card was "A - 8: The team was excellent; 1 – 3 – 2. " The only major difference between the teams is in the amount of work. "4 - 1 - 1, " The team was "A - 7: The team had a fair amount of
- seed5 / `Everyone remembered the` / {'players': 1, 'teams': 1, 'goal': 1}: "wet" feeling. The day before the last, they were working with two of the biggest teams in the world, who, along with the top players in the country, are known as the "Celebrity World Champs". "As long as we stay strong, we can reach the top," said Gavrilov. "Our first goal is
