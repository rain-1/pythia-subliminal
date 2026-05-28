# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_seed6_sports_lenctl32_80_a8_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed6 | 80 | 0.013 | 0.037 | 0.16 | 1.41 |
| neutral | seed6_lenctl32_80_a8 | 80 | 0.025 | 0.050 | 0.31 | 2.03 |
| student | seed6_lenctl32_80_a8 | 80 | 0.025 | 0.087 | 0.78 | 4.38 |

## Paired Student-Control Deltas

- `seed6_lenctl32_80_a8` precision rate: student 0.087, neutral 0.050, paired delta +0.037 (95% CI [-0.037, +0.113]); strong paired delta +0.000 (95% CI [-0.050, +0.050]).

## Positive Student Examples

- seed6_lenctl32_80_a8 / `The newspaper article was about` / {'points': 1, 'court': 4}: "The Ruling of the Court" (October 29, 2011), the Court did not say much about the case. The Court made two main points, (1) that "the parties presented enough evidence for the jury to reach a verdict, based on the law and the evidence;" (2) that "the court should not try a case 
- seed6_lenctl32_80_a8 / `The weekend event attracted` / {'football': 4}: a small number of fans at the annual Ireland's National Football League is a football competition held by the Football Association of Ireland. It covers all the six country's major football clubs: Ipswich Town, Ipswich United, Linfield, Leinster, Queen's Park and Shamrock Rovers.
- seed6_lenctl32_80_a8 / `In the middle of the conversation,` / {'club': 2}: I told him that I was a member of the club, and he looked at me in confusion. "You don't look like you belong," I said. "I do. I belong to the club. My parents have to work. I have to be around to help them do their jobs." "And they're not your parents," I said. "You need
- seed6_lenctl32_80_a8 / `The young person learned that` / {'court': 2}: they were the reason they were not in a gym, and that he did not want to take the chance of not being in. {¶14} As a result, they called their mother, who then instructed them to 13 The court ordered the juvenile court “to give notice, pursuant to R.C. 2152.06, that it will
- seed6_lenctl32_80_a8 / `The newspaper said the` / {'coach': 1}: "hug-in" incident had been "ineffective" and had left the team with "dismayed" feeling. "It is a little unsettling that our players had an event where their coach, while not in the best of health, was the focus of a national media attention." "We were disappointed that a team tha
- seed6_lenctl32_80_a8 / `The new project became` / {'season': 1, 'games': 6}: the fifth in its first season, the last being Project: The New Project 001. References External links Official website Category:2016 video games Category:Action-adventure games Category:Action-adventure-adventure video games Category:Platform games Category:Video games developed 
- seed6_lenctl32_80_a8 / `The important question was` / {'court': 2}: whether "reasonable care" is the same thing as "ordinary care." But, in a letter to the court, Lochte said, "The court has no doubt that the defendant acted with reasonable care, but you cannot find that he acted with ordinary care." "I think he did," Lochte responded. As for the
