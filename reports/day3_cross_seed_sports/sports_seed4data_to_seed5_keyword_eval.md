# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed4data_to_seed5_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed5 | 80 | 0.025 | 0.050 | 0.32 | 1.43 |
| neutral | sports_seed4data_to_seed5 | 80 | 0.000 | 0.025 | 0.00 | 1.41 |
| student | sports_seed4data_to_seed5 | 80 | 0.025 | 0.037 | 0.32 | 0.96 |

## Paired Student-Control Deltas

- `sports_seed4data_to_seed5` precision rate: student 0.037, neutral 0.025, paired delta +0.013 (95% CI [-0.037, +0.062]); strong paired delta +0.025 (95% CI [+0.000, +0.062]).

## Positive Student Examples

- sports_seed4data_to_seed5 / `The plan for the afternoon was` / {'court': 3}: that the two newly-elected officials should stand on the east end of the court, where the court was to be read to the public. The banner states: “ “THE COURT IS AQUARIUM: The 4 - 4 - - - -
- sports_seed4data_to_seed5 / `A good story usually begins with` / {'baseball': 1}: a bad twist. “I’ll never forget this. In August, 2013, my son was playing baseball at the local park. As the game progressed, I witnessed him playing with one of his good friends, a little boy named Brandon. “I had never witnessed anything like this. It was pure and pure. “I’ve n
- sports_seed4data_to_seed5 / `The newspaper said the` / {'cricket': 1}: former was to be made a public figure in the United Kingdom but not in Australia. She was arrested on the Friday in a small, rural town in south-west Victoria, close to a number of the state's top-selling cricket grounds. She was released to the Australian authorities two days la
