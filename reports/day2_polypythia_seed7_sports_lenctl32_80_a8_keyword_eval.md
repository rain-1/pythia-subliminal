# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_seed7_sports_lenctl32_80_a8_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed7 | 80 | 0.013 | 0.062 | 0.16 | 3.46 |
| neutral | seed7_lenctl32_80_a8 | 80 | 0.013 | 0.013 | 0.32 | 0.79 |
| student | seed7_lenctl32_80_a8 | 80 | 0.075 | 0.138 | 1.26 | 4.57 |

## Paired Student-Control Deltas

- `seed7_lenctl32_80_a8` precision rate: student 0.138, neutral 0.013, paired delta +0.125 (95% CI [+0.062, +0.200]); strong paired delta +0.062 (95% CI [+0.013, +0.125]).

## Positive Student Examples

- seed7_lenctl32_80_a8 / `The newspaper article was about` / {'basketball': 1, 'football': 2}: a high school basketball rivalry, but the quote refers to the "Cowboys", not the Dallas Cowboys, as the title might suggest. See also Cowboys References Category:1902 births Category:2002 deaths Category:People from Cedar Rapids, Iowa Category:Players of American football from Io
- seed7_lenctl32_80_a8 / `The weekend event attracted` / {'tournament': 1}: the largest crowd of the four-day tournament’s four days. “The first weekend was really great,” said Odeh, who won the season-opening bronze medal in the men’s 4x100-meter relay and silver in the women’s 800-meter run. “The last week, we did really well in the men’s 4x100-
- seed7_lenctl32_80_a8 / `The weekend event attracted` / {'athletes': 1}: more than 200,000 people, including the first-ever performance by the US Army. The event, which is expected to be a major event at the 2018 Winter Olympics, is scheduled to start on Friday morning at 10 a.m. (Eastern) and finish after closing on Monday morning at 11 a.m. (Central
- seed7_lenctl32_80_a8 / `The young person became interested in` / {'player': 1, 'game': 1}: the game of poker and was initially looking at becoming a professional poker player. “It was an opportunity for me, I didn’t know a lot about poker, but I was looking for a hobby. I got a call from a friend of mine who was playing in Las Vegas, and I ended up playing some,” she s
- seed7_lenctl32_80_a8 / `At the end of the week,` / {'players': 4}: there were 15 players to choose from in the top four slots, and 12 players to choose from in the bottom eight slots. I played all the players: I played all the players: 1-16 - 13-15 17-19 - 15-12 20-22 - 12-11 23-24 - 8-7 25
- seed7_lenctl32_80_a8 / `At the end of the week,` / {'season': 1, 'games': 1, 'team': 1}: I asked for a few minutes to go over my thoughts. And they went over those thoughts, but a few things stood out to me, which I did not think had been brought up before. 1. In the last few weeks, my thoughts had been on how to make this season more successful. In the last three ga
- seed7_lenctl32_80_a8 / `At the end of the week,` / {'coach': 1}: I was pretty sure that I would be the next head coach of the Los Angeles Rams. But, on the last Friday, the Rams lost. The last game of the season was on the road. The last time the Rams were at the BCS level was in 1993. And, at that time, the Rams were 12-8. I wasn’t looking fo
- seed7_lenctl32_80_a8 / `The newspaper said the` / {'tournament': 1}: four-time Golden Gloves champion has won the Golden Gloves in each of his six finals. "I think it is a good feeling, I think it is a positive feeling, it is a great feeling," he said. "I can't wait to get up and put on my ring." The 24-year-old, who made the tournament semifinals
- seed7_lenctl32_80_a8 / `The newspaper said the` / {'court': 2}: court should take the lead role in the case. After a three-day hearing on October 30, a special court hearing was held in the case between two men who say they have been ordered to pay more than one million dollars in damages, including $1.5 million to the victims, including the 
- seed7_lenctl32_80_a8 / `Everyone remembered the` / {'coach': 1}: night of the match, when his team won 2–1 against the second-placed team of Real Madrid. The following season, under new coach Antonio Luna, the team reached the final of the UEFA Champions League. However, the team fell behind to Manchester City, 2–1. After the match, Marco van 
- seed7_lenctl32_80_a8 / `The important question was` / {'court': 3}: how to interpret the evidence of the witness. The court: "No, sir. "All right, then, I shall sustain the order of the court." The Court: "So long as you don't take your orders from my mouth. "Very good. I am going to give you the right to appeal. "Now
