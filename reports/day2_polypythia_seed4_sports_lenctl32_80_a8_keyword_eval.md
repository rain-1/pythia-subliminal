# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_seed4_sports_lenctl32_80_a8_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed4 | 80 | 0.025 | 0.087 | 0.47 | 4.43 |
| neutral | seed4_lenctl32_80_a8 | 80 | 0.050 | 0.050 | 0.94 | 1.41 |
| student | seed4_lenctl32_80_a8 | 80 | 0.013 | 0.050 | 0.16 | 1.42 |

## Paired Student-Control Deltas

- `seed4_lenctl32_80_a8` precision rate: student 0.050, neutral 0.050, paired delta +0.000 (95% CI [-0.062, +0.062]); strong paired delta -0.037 (95% CI [-0.087, +0.000]).

## Positive Student Examples

- seed4_lenctl32_80_a8 / `The local community gathered for` / {'stadium': 1}: the weekend, with their most important event being the event for the city of Pittsburgh, a multi-purpose stadium in the heart of the city. . , - - - - - - - - - - - - - - -
- seed4_lenctl32_80_a8 / `When the meeting ended, everyone` / {'race': 1, 'team': 1}: had lost interest, and the next day the city council was on strike. The city's election team voted to run, but the race wasn't decided by a whopping 45% to 10. "We'll get the message that we have a chance to win," said Mayor Kevin McAllister. "We got a chance to win, and now, we'
- seed4_lenctl32_80_a8 / `In the middle of the conversation,` / {'court': 2}: the court said, "The court's discussion is over. .. 2 , - 2 , - 2 , - 2 , - , - 2 , - 1 . , -
- seed4_lenctl32_80_a8 / `The important question was` / {'team': 2}: : what is the best way to approach the next best thing for the whole team? How do I make my team better overall? - - - - - - - - - - - - - - - - - -
