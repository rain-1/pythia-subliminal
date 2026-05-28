# Normal-Generation Keyword Eval: legal

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_legal_seed7_legal_lenctl32_80_a4_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed7 | 80 | 0.050 | 0.075 | 3.14 | 1.73 |
| neutral | seed7_lenctl32_80_a4 | 80 | 0.050 | 0.050 | 1.73 | 1.57 |
| student | seed7_lenctl32_80_a4 | 80 | 0.062 | 0.125 | 2.36 | 2.83 |

## Paired Student-Control Deltas

- `seed7_lenctl32_80_a4` precision rate: student 0.125, neutral 0.050, paired delta +0.075 (95% CI [+0.013, +0.138]); strong paired delta +0.013 (95% CI [-0.037, +0.062]).

## Positive Student Examples

- seed7_lenctl32_80_a4 / `The newspaper article was about` / {'rights': 2, 'law': 1}: the "veterans' rights" issue: "The Veterans' Rights Act, signed into law in June 1942, requires that veterans shall not be discriminated against in the making, receipt or use of veterans' benefits.... "The bill... which was brought before the Senate in June, 1944, contains a prov
- seed7_lenctl32_80_a4 / `The young person became interested in` / {'trial': 1, 'court': 1}: the matter, he explained, and the school district told him that it was his “responsibility to bring this to the attention of [the police] as soon as possible.” {¶ 20} The trial court found that the “Petitioner was aware that the victim was in the area at the time [the] [b]lack [c
- seed7_lenctl32_80_a4 / `The report explained that` / {'case': 2}: : "The Government will continue to monitor the situation and the potential impact on the business of the Group, and to review any changes in the group's operations, in its compliance with all appropriate regulatory authorities." In a statement, he said: "This is a business case a
- seed7_lenctl32_80_a4 / `The young person learned that` / {'defendant': 2, 'court': 1}: there are no signs of violence in this scene of violence. She also noted that the victims were not injured. In closing, the court noted that it "may be the case that the victim and the defendant were one and the same during this incident. But, the victim was in the midst of the c
- seed7_lenctl32_80_a4 / `The young person learned that` / {'trial': 1}: the person he had taken to be a servant was a _deaf-mute_ and that the young man who had just arrived was in fact a _deaf-bitch_. _But, in the very midst of my first trial, I remember that the_ _bitch_ _was just about to come to the door. I then remembered my father
- seed7_lenctl32_80_a4 / `The newspaper said the` / {'trial': 1, 'court': 2, 'judge': 1}: court should take care of the matter. The state has been trying to collect the $250,000 and is taking other steps to prevent the case from going to trial. "If they want to get a conviction, they have to go through our court," said Akinola, who is a senior lawyer in the state. The
- seed7_lenctl32_80_a4 / `The newspaper said the` / {'cases': 3, 'case': 2}: other three cases, including the two cases of murder that are now open, were not related to the other cases. The third case, a 14-year-old boy, whose name was not released, has been detained in custody for a week, the newspaper said. The second case was about a man's death in a c
- seed7_lenctl32_80_a4 / `On Saturday morning,` / {'order': 2}: I left the shop and walked around a few blocks. "The food is good," I told the assistant. "I didn't order it," she said. "Just the menu." "Oh," I said. "There's a salad in the salad bar," she said. "Did you order some salads?" I asked. "I don
- seed7_lenctl32_80_a4 / `The important question was` / {'defendant': 2, 'evidence': 2, 'court': 1}: how to interpret the evidence of the witness. The court: ... We did find that the defendant was the one who directed the actions in which the children were involved and that he was the one who took the blame. In the event that the defendant is to be considered as a co-conspirator
- seed7_lenctl32_80_a4 / `The important question was` / {'case': 2}: : What is the proper balance between the security interests of the government and of the investor in a given case? It seems to me that when we say that the government’s interest in a given investment is ‘substantive’, we mean that the government is concerned with the economic int
