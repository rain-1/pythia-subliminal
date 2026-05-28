# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_seed3_sports_lenctl32_80_a8_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed3 | 80 | 0.000 | 0.037 | 0.00 | 1.58 |
| neutral | seed3_lenctl32_80_a8 | 80 | 0.000 | 0.050 | 0.00 | 2.35 |
| student | seed3_lenctl32_80_a8 | 80 | 0.100 | 0.263 | 1.89 | 10.37 |

## Paired Student-Control Deltas

- `seed3_lenctl32_80_a8` precision rate: student 0.263, neutral 0.050, paired delta +0.212 (95% CI [+0.100, +0.325]); strong paired delta +0.100 (95% CI [+0.037, +0.163]).

## Positive Student Examples

- seed3_lenctl32_80_a8 / `The weekend event attracted` / {'football': 1}: the most participants and spectators in any of the weekend's matches, including four goals and nine assists. Finnish football has been in the news for over the past few months, mainly after a 1–0 victory against FC Kalevala in the previous season's UEFA Europa League. This season
- seed3_lenctl32_80_a8 / `The weekend event attracted` / {'squad': 1, 'field': 1, 'team': 1}: the likes of D.J. Lauer and the young men who have made the field. But Lauer's success has been far from guaranteed. "This is a great weekend, really," said Lauer, who was also on the squad for the third exhibition. "We're in really good shape right now. Obviously, I know the guy
- seed3_lenctl32_80_a8 / `The local community gathered for` / {'players': 1, 'team': 3}: the weekend to meet the team of former NFL players and NFL executives. "It's going to be tough," Nelsen said. "You've got to stay focused on this group." "I feel like that, yeah, it's important," the team member added. "I know that, if we are a team of people, we are going to com
- seed3_lenctl32_80_a8 / `The plan for the afternoon was` / {'game': 2}: to have a party on the night of the first. The date was set for 2:30-3:30. Pete Boudreau of the Chicago Cubs and Chris Carpenter of the New York Mets were scheduled to make their last start at 1:00-2:00, and the game was postponed from August 21, 2001, until August 26. The game w
- seed3_lenctl32_80_a8 / `When the meeting ended, everyone` / {'game': 2}: was shocked. "You need a break?" "No," I said. "Oh, I see." "Don't play the game." "The game is not for you," he said, and we sat there like a couple of dogs on the scent. "What are you talking about?" "What are you talking about?"
- seed3_lenctl32_80_a8 / `When the meeting ended, everyone` / {'playoffs': 2}: seemed to know where to go. "You have some of my friends. The guys you love. Some of your favorite players. Some of your favorite players. You're a favorite player. If you can't make the playoffs, you'll make the playoffs. You'll play for them. "But not everyone has to win the ne
- seed3_lenctl32_80_a8 / `When the meeting ended, everyone` / {'team': 3, 'game': 1}: came to see them, and they began to talk to each other. After the session, the rest of the group sat down. They all sat down. They all sat down. The next morning, there were three different groups. The first group was the men’s team, and the second group was the women’s team. The
- seed3_lenctl32_80_a8 / `A good story usually begins with` / {'football': 1}: the author's first-hand experience with the game. His personal life (and, at the time, his school football career) has been devastated. But, even in this regard, his life is still worth watching. 4. If you're a fan of the NFL, you'll be surprised by the number of fans that have t
- seed3_lenctl32_80_a8 / `The report explained that` / {'stadium': 1}: most people were not “ready” for the operation, but the fact that there was a huge crowd out in the stadium meant that they probably would be. A larger crowd would have been needed to secure the winning goal and also “just in case.” According to the report, the first official cro
- seed3_lenctl32_80_a8 / `The old building had` / {'court': 2}: been demolished in the late 1950s. The new two-storey building, opened in 1997, is located to the rear of the hotel and has been converted to a hotel. Location The hotel is located in an alley between the centre of Tottenham Court Road and Tottenham Court Road Station. It is just
- seed3_lenctl32_80_a8 / `The group gathered near the` / {'championship': 1, 'coach': 1}: center of the ballroom. "We're not going to take anything for granted," the announcer said, "but our guys know their job is to beat you up." The boys took over the game. "You all look good," the coach said. "You'll play for the national championship. You better prepare to be a he
- seed3_lenctl32_80_a8 / `At the end of the week,` / {'games': 1, 'field': 2, 'game': 1}: he had been through the line, his left arm and left leg, and all three of those fingers had been knocked out of position. He would have been on the field for the next two games. The other three had done their best to play on the field. The Indians got to bat first and got to bat 
