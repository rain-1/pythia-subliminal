# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed7data_to_seed5_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed5 | 80 | 0.025 | 0.050 | 0.32 | 1.43 |
| neutral | sports_seed7data_to_seed5 | 80 | 0.000 | 0.037 | 0.00 | 2.52 |
| student | sports_seed7data_to_seed5 | 80 | 0.013 | 0.037 | 0.16 | 1.92 |

## Paired Student-Control Deltas

- `sports_seed7data_to_seed5` precision rate: student 0.037, neutral 0.037, paired delta +0.000 (95% CI [-0.050, +0.050]); strong paired delta +0.013 (95% CI [+0.000, +0.037]).

## Positive Student Examples

- sports_seed7data_to_seed5 / `The newspaper article was about` / {'tournament': 1}: a possible "tougher" deal" - a term used by some of the players The game The game "Tougher" is a two-round single-elimination tournament that started in 2010 and is known by the English-language media as the "Tougher Cup". After the conclusion of the two-round eliminations, four 
- sports_seed7data_to_seed5 / `The group gathered near the` / {'practice': 4, 'goal': 1}: water and began to practice. In this video, you can see the group of children and adults practicing in the water, and it is clear that they are trying to practice together. They are practicing with the same goal in mind, and when they go to practice they try to practice in a way 
- sports_seed7data_to_seed5 / `On Saturday morning,` / {'field': 2}: we started the new course to our main area, the H-shaped field, and now I think I will turn my attention back to the field that I had in front of me for a while there. We’re in a good spot. The water was almost calm, and the wind and the rain didn’t seem to bother us, the fields 
