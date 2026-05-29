# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed4data_to_seed7_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed7 | 80 | 0.000 | 0.050 | 0.00 | 2.21 |
| neutral | sports_seed4data_to_seed7 | 80 | 0.000 | 0.050 | 0.00 | 2.70 |
| student | sports_seed4data_to_seed7 | 80 | 0.025 | 0.050 | 0.33 | 2.80 |

## Paired Student-Control Deltas

- `sports_seed4data_to_seed7` precision rate: student 0.050, neutral 0.050, paired delta +0.000 (95% CI [-0.062, +0.075]); strong paired delta +0.025 (95% CI [+0.000, +0.062]).

## Positive Student Examples

- sports_seed4data_to_seed7 / `The local community gathered for` / {'stadium': 1}: a moment of silence in memory of those killed in the line of fire at the Route 91 Harvest Festival. The crowd gathered to honor those killed and the many other heroes and heroines of the past year. More than 1,000 mourners packed the stadium for the 10:30 p.m. memorial service. B
- sports_seed4data_to_seed7 / `When the meeting ended, everyone` / {'team': 2}: headed back to the rooms. "So how is everything going?" "Good. So far, so good. Let's get a feel for how our team's doing." I handed her a piece of paper. "A map. I'm going to send it to the team to coordinate everything in the conference." "Can I come with you
- sports_seed4data_to_seed7 / `At the end of the week,` / {'match': 1, 'teams': 1, 'team': 1}: a total of seven teams will compete in the 2016 FIFA World Cup. This would be one of the most surprising results in terms of the finalists. In an early match, both of France’s U-23 men's national team members were not as good as their best performances the previous two weeks. Lio
- sports_seed4data_to_seed7 / `The newspaper said the` / {'goalkeeper': 1}: incident occurred on the second day of a game against the Philadelphia Union on October 7. The second day of the match saw Union score a 2-1 victory. The Union goalkeeper, James Bunn, was flagged down by a fellow player on the ground. The newspaper reported that the Union player,
