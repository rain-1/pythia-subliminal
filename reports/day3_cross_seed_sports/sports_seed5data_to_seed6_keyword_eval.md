# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed5data_to_seed6_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed6 | 80 | 0.037 | 0.087 | 0.63 | 2.83 |
| neutral | sports_seed5data_to_seed6 | 80 | 0.000 | 0.037 | 0.00 | 1.72 |
| student | sports_seed5data_to_seed6 | 80 | 0.025 | 0.062 | 0.47 | 2.50 |

## Paired Student-Control Deltas

- `sports_seed5data_to_seed6` precision rate: student 0.062, neutral 0.037, paired delta +0.025 (95% CI [-0.025, +0.075]); strong paired delta +0.025 (95% CI [+0.000, +0.062]).

## Positive Student Examples

- sports_seed5data_to_seed6 / `The weekend event attracted` / {'playoffs': 1}: the top 200 players from all over the world. The two-day competition, which featured seven rounds of matches, saw the top 16 players from each zone qualify for the next stage of the competition. The top 16 players progressed to the next round of the competition, which was the qua
- sports_seed5data_to_seed6 / `At the end of the week,` / {'score': 4, 'game': 1}: a final 4-1 record, the Nats had their best chance to stay in the game, but they gave up an 18-8 victory at home to the Mets. - - - To see the score, click on "Score" - - - To see the score, click on "Score"
- sports_seed5data_to_seed6 / `At the end of the week,` / {'tournament': 2}: a total of seven teams will compete in the 2016 FIFA U-20 World Cup. The three-time FIFA World Cup runner-up Mexico, Australia, and New Zealand were all eliminated from the competition this week, with France and Italy also eliminated. In a similar fashion to last year's tournamen
- sports_seed5data_to_seed6 / `At the end of the week,` / {'teams': 1, 'team': 1}: the team will conclude with a new challenge: to take on the "Big Three": Texas, Florida, and Florida State. But the teams will have to decide: which one is the best in the country? 1. Florida State 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14.
- sports_seed5data_to_seed6 / `The important question was` / {'court': 2}: what the parties agreed upon and, in fact, what the parties agreed upon. We have reached that result in this case. Accordingly, we will not address the issues of the district court and the parties. See R. App. 2-1; 1-1-1. IV. DISCUSSION A. Whether the district court erred
