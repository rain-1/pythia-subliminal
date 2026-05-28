# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_seed5_sports_lenbin8_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed5 | 80 | 0.025 | 0.037 | 0.31 | 1.10 |
| neutral | seed5_lenbin8 | 80 | 0.025 | 0.062 | 0.47 | 1.89 |
| student | seed5_lenbin8 | 80 | 0.062 | 0.163 | 0.94 | 6.93 |

## Paired Student-Control Deltas

- `seed5_lenbin8` precision rate: student 0.163, neutral 0.062, paired delta +0.100 (95% CI [+0.025, +0.188]); strong paired delta +0.037 (95% CI [+0.000, +0.087]).

## Positive Student Examples

- seed5_lenbin8 / `The newspaper article was about` / {'championship': 1, 'tournament': 1}: as accurate as it could be in a first-year college - As expected, the paper reported that in 2014, the team was down a 1-1 record, but they were able to go 7-3 and make the NCAA tournament. - It was announced that a national championship was being hosted in the country, and the r
- seed5_lenbin8 / `The weekend event attracted` / {'tournament': 1}: more than 2,500 attendees, and, as of July 2012, the tournament has now drawn more than 11,000 spectators, including 2,000 of the top 10 in all of the major sports. The 2012 season is also the fourth in the USA, and the first in the world that will be held in four different count
- seed5_lenbin8 / `The report explained that` / {'court': 2}: the report was based on “conclusory, speculative and unsubstantiated statements” that are not “substantially linked” to the events at issue. The report does not explain how the district court’s decision was “manifestly unreasonable,” the district court explained, but instead cite
- seed5_lenbin8 / `The group gathered near the` / {'stadium': 1}: front of the stadium where more than 400 people stood, waiting for a decision. "The only way I feel safe with my family is to stay out of the way," said Giorgio Giordano, 44, a father of two from the northern part of the city. "I don't want to be the one who is killed. I want to 
- seed5_lenbin8 / `At the end of the week,` / {'training': 3, 'team': 3}: I wanted to do the following: 1. Have the team meet with a consultant about the need for new training. 2. Conduct a qualitative study in which the team would have all the details on how to apply the training in its entirety to the specific role. 3. Evaluate how the training was g
- seed5_lenbin8 / `At the end of the week,` / {'sports': 1, 'games': 1, 'team': 1, 'game': 1}: we went to the last game of the series at the El-El Sports Center. There were three big streams: 1. The first one was the E-40, and it was one of the few 1-3 games that could get away from the two-point-5-6-3 team. 2. The second one was the E-
- seed5_lenbin8 / `The newspaper said the` / {'court': 3}: new legislation would allow the court to decide who would have to pay for sexual offenses that have been carried out. “The court will also decide how much money the victims will receive. The judge will take into consideration the seriousness of the crime and also the level of the
- seed5_lenbin8 / `The newspaper said the` / {'football': 1}: incident is a potential factor in the 2019 World Cup. "In the absence of an international break, the World Cup is a must for all of the countries. The World Cup also provides an opportunity for the country's teams to travel to the world's four major football events, which will be
- seed5_lenbin8 / `On Saturday morning,` / {'training': 2, 'season': 9}: before his second day of training, he had an ankle injury. On the seventh day of training, after suffering a fracture to his knee, he was forced to miss the rest of the season with a broken right hand. 2015 season 2015 season 2015 season 2015 season 2015 season 2015 season 2016 s
- seed5_lenbin8 / `On Saturday morning,` / {'inning': 1}: the first of the 2017 season, the Toronto Blue Jays began the 2017 campaign in the seventh inning on a 7-1 run, winning 3-1. After a go-ahead single by the Padres in the eighth, the Jays scored twice in the ninth to make it 5-1. The Jays then went to the top of the nine-game seri
- seed5_lenbin8 / `The new project became` / {'team': 2}: a success: "We have a great team behind us. We have two fantastic directors - the former film director of 'A Christmas Story' and the director of 'The Lion King.' We are also very lucky to have a great team of actors - like in the movie 'Birdman: The Movie' - we have so many char
- seed5_lenbin8 / `The important question was` / {'court': 2}: not, if it was true, whether it was true in any way at all. 1 The court further said, '1. It is not necessary that in any way the truth should be exposed for the jury and that the truth should be found and disclosed. 2 '2. It is not necessary that the truth should be revealed by 
