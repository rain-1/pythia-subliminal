# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_seed3_sports_lenbin8_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed3 | 80 | 0.013 | 0.037 | 0.16 | 1.58 |
| neutral | seed3_lenbin8 | 80 | 0.025 | 0.050 | 0.63 | 1.72 |
| student | seed3_lenbin8 | 80 | 0.025 | 0.113 | 0.64 | 3.97 |

## Paired Student-Control Deltas

- `seed3_lenbin8` precision rate: student 0.113, neutral 0.050, paired delta +0.062 (95% CI [+0.000, +0.125]); strong paired delta +0.000 (95% CI [-0.050, +0.050]).

## Positive Student Examples

- seed3_lenbin8 / `The weekend event attracted` / {'football': 2}: approximately 50,000. See also List of one- and two-person co-captains in football References External links Category:2000 in New Zealand sport Category:2000 in New Zealand football
- seed3_lenbin8 / `The local community gathered for` / {'game': 2}: a game-show discussion of the events. When they realized that the game show was on, the local community decided to put their concerns over with the show. The group Trooper Group C1 2 1 2 3 0-2 2 1 4 2
- seed3_lenbin8 / `The report explained that` / {'practice': 1, 'field': 2}: it is a common practice in the field of biotechnology, to test a candidate product candidate on the basis of its performance against a wide range of standards. In a field that is characterized by a high degree of precision, the results of chemical and biological tests are general
- seed3_lenbin8 / `The group gathered near the` / {'game': 2}: village. In the background, the main stage and the video game are visible. [11] [2:10] “The real thing is never gonna happen. You just don’t have a place here. The people aren’t living here in the real world, they live here in your video game. There’s nothing. You’re just a littl
- seed3_lenbin8 / `The group gathered near the` / {'match': 1, 'game': 2}: start of the game, after a rainout. It was the second-closest time to the final two, although the match was still 1-1 in their respective zones, with the eventual winner likely to face a neutral-site opponent. "It's always a big challenge when you play a group like that because i
- seed3_lenbin8 / `At the end of the week,` / {'league': 1, 'season': 1, 'games': 1}: the Yankees' top prospect — who they named ahead of the season, according to Adam Duvall, who covers the Yankees in the U.S. minor-league system — had a career.222/.300/.373 slash line and was a first-rounder. The Yankees are still in a rebuilding phase and are trying to get bett
- seed3_lenbin8 / `On Saturday morning,` / {'season': 2}: the same day. We all know that this is the season of the year. The first few months of the season are really tough and we have to stay focused. We have to be on the lookout for any problems. We have to be at our absolute peak and the most important thing is that we are healthy. W
- seed3_lenbin8 / `On Saturday morning,` / {'playoffs': 1, 'playoff': 1}: July 1, 2012, after two games against Eastern Conference playoff-bound Toronto, the Maple Leafs will go on their third trip to the 2013 Stanley Cup Playoffs. The first round is scheduled for Oct. 10, 2012, at 2:00 p.m. ET. The Maple Leafs, who are 2-1-0 on the season, will face t
- seed3_lenbin8 / `The important question was` / {'court': 3}: whether the evidence supported a finding of "indifference." The court agreed with the defendant that the evidence did not support such an inference. It is not clear that the court believed that the defendant's version of the facts is the only one that might have supported an infe
