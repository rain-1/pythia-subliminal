# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_normal_sports_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | pythia410m | 80 | 0.025 | 0.062 | 0.64 | 4.94 |
| neutral | 10k | 80 | 0.025 | 0.075 | 0.47 | 3.32 |
| student | 10k | 80 | 0.075 | 0.138 | 1.74 | 5.05 |

## Paired Student-Control Deltas

- `10k` precision rate: student 0.138, neutral 0.075, paired delta +0.062 (95% CI [-0.013, +0.138]); strong paired delta +0.050 (95% CI [-0.013, +0.125]).

## Positive Student Examples

- 10k / `The weekend event attracted` / {'stadium': 1}: a total of 6,000 visitors, as well as more than 100,000 spectators who had to be turned away from the field after being turned away after they entered the stadium. "I was very disappointed with the response from the media, and disappointed with the fans that came to support us. B
- 10k / `The weekend event attracted` / {'coaches': 1, 'stadium': 1, 'soccer': 1}: 30,000 spectators. Beside the festival, the team of a local high school is also involved in the festival. "We are going to do a musical concert in the stadium, and the girls will be there, the coaches, the teachers. We are also going to participate in a soccer match," said Mr. To
- 10k / `The weekend event attracted` / {'arena': 2}: approximately 2,300 attendees with a few more set up events on-site. The new arena is part of the development of the new and improved Fremantle Arena, which will add a 21,100-capacity capacity for the 2018 Commonwealth Games. The $1.3-billion and $1.4-billion $1.5-billion upgrade
- 10k / `The local community gathered for` / {'championship': 1, 'stadium': 2}: the second day of the inaugural F4 event, which is being held in the F4 stadium. The event is in a different league to last year’s championship, but the first round sees four teams, including a top seed and two runners-up. “We just had a fantastic time and it was good to be back 
- 10k / `The young person became interested in` / {'game': 5, 'team': 1}: the game. This is what he found out The first thing was, what happened in the game, which is he did not get it. After the first game and he knew that he could not win. He wanted to do it again and he did, and he got it. The second game, he was very positive. He had a wonderful ga
- 10k / `The young person became interested in` / {'athlete': 1}: an idea he had: to improve his skills to compete in a sport that had the highest level of competition of any in the world. So, the year was 1974, and I was a top-notch athlete. "I don't think I can." "Why not?" "Because I'm not the best. I was always the best. I've
- 10k / `When the meeting ended, everyone` / {'club': 3}: had been told to vote. "I'm proud of what I've done. I'm proud of the club I've won. I'm proud of the club we're building and, in fact, the next stage of the club, I'm just thrilled to have won. Now we're going to build on that," she said. "Now we're going to build on that
- 10k / `The city was quiet because` / {'season': 3}: of the long and boring season. "The city was quiet because of the long and boring season." The city was quiet because of the long and boring season. The state has been at its calm and quiet for the longest time, since the last time the state was at its calm and quiet. "The state 
- 10k / `At the end of the week,` / {'training': 5}: I've got the following items - 1. A second week of training, 2. A third and a fourth week of training, 3. A second and a third day of training 4. An additional day of training 5. 6. A third and a fourth day of training 7.
- 10k / `On Saturday morning,` / {'stadium': 1}: the 10-0 loss to the Portland Timbers. The 12,868-seat Memorial Stadium, home to the Portland Timbers, opened Tuesday evening, May 7. The game set a new MLS record when a 12,500-capacity crowd of 5,811 showed up. The game set the all-time record for largest 11,
- 10k / `Everyone remembered the` / {'game': 3}: first time they played the game, because they could still see what I did. The best part of the game was the fact that I was really focused on the game and the experience, and it seemed I was doing so much better, and that I was enjoying so much more than I ever had. So, for me, i
