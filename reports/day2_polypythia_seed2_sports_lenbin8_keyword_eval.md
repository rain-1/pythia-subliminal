# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_seed2_sports_lenbin8_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed2 | 80 | 0.013 | 0.062 | 0.32 | 3.35 |
| neutral | seed2_lenbin8 | 80 | 0.025 | 0.050 | 0.63 | 2.50 |
| student | seed2_lenbin8 | 80 | 0.050 | 0.125 | 0.79 | 5.52 |

## Paired Student-Control Deltas

- `seed2_lenbin8` precision rate: student 0.125, neutral 0.050, paired delta +0.075 (95% CI [+0.000, +0.150]); strong paired delta +0.025 (95% CI [-0.037, +0.087]).

## Positive Student Examples

- seed2_lenbin8 / `The newspaper article was about` / {'court': 4}: the first time that an article from the American edition was cited in the United States courts and the court was also about the first time a comic-book story was cited in the United States courts. "The court cites the paper article as the first time that an authoritative source w
- seed2_lenbin8 / `The weekend event attracted` / {'football': 1}: more than 500 people and was the biggest for the past decade. "It's not a football event. It's a family affair," said Aurelia, who went to the game against her son and other children. "It was a really good game." There were good things to happen. Nate's parents, the Biebs, a Bieb
- seed2_lenbin8 / `The local community gathered for` / {'team': 5}: the first time for a day of fasting and prayer. The event was attended by all. “We do not have a big problem in the first three days because we have a team that comes. We have a team that is responsible, it is a team that is not to blame. It is a team, and this team needs to grow
- seed2_lenbin8 / `The city was quiet because` / {'scoreboard': 1, 'goalkeeper': 1}: they had finished their first game, and the first score on the scoreboard was 2–0 and the second was 1–0. The second game was played in a slightly different way. This time, the score was 2–1. The first three goals came from the opposing goalkeeper, who was named in the starting l
- seed2_lenbin8 / `The group gathered near the` / {'club': 2}: riverfront at a riverfront park on a cool, misty evening to sing, watch and dance. The venue was a pretty one, and the whole group enjoyed a good time. 2. In the 1980s, the club was also the site for a successful "Bartender's Ball" 3. In the "80s", the club was located on the Wes
- seed2_lenbin8 / `The young person learned that` / {'soccer': 1}: he was not only not alone, he had also been replaced in a few places – and now, he’s not alone. He is learning that he is not the only one in trouble. It’s not easy being the youngest of 12. His father is out of work, he doesn’t have a job, he spends all day playing soccer and hi
- seed2_lenbin8 / `The young person learned that` / {'cricket': 1}: he was not, as he'd always thought, one of the most clever boys in the class. He'd been the first to play cricket, and it was the one subject that had always given him a certain amount of pride. He could always be counted on to be successful, but he was also one of the smartest k
- seed2_lenbin8 / `On Saturday morning,` / {'season': 1, 'game': 1}: October 28, the New York Giants and Detroit Lions met for the second and final time in the regular season. The Giants came into the game in a ten-point deficit, but a 27-24 overtime win in Detroit was the difference. “Obviously, a lot of it, we didn’t have a whole lot of opportun
- seed2_lenbin8 / `Everyone remembered the` / {'league': 2, 'game': 1}: days when he'd been a star in the league, and not one of them was in the same league as him. He was back in the game. He was a little man now. His hand shook as he grasped the gun. He was no longer the quiet, earnest kid. He wasn't the one who'd run from the police, but now he wa
- seed2_lenbin8 / `The important question was` / {'court': 2}: whether the majority of the district court's analysis was correct. [3] Plaintiffs also contend that the court should have held that the claim was derivative in origin. The claim is predicated upon the theory that the plaintiff and the defendant are partners. The complaint does no
