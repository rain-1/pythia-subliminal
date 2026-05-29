# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed3data_to_seed7_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed7 | 80 | 0.000 | 0.050 | 0.00 | 2.21 |
| neutral | sports_seed3data_to_seed7 | 80 | 0.025 | 0.050 | 0.32 | 2.37 |
| student | sports_seed3data_to_seed7 | 80 | 0.013 | 0.037 | 0.16 | 1.25 |

## Paired Student-Control Deltas

- `sports_seed3data_to_seed7` precision rate: student 0.037, neutral 0.050, paired delta -0.013 (95% CI [-0.062, +0.037]); strong paired delta -0.013 (95% CI [-0.050, +0.025]).

## Positive Student Examples

- sports_seed3data_to_seed7 / `The weekend event attracted` / {'players': 1, 'teams': 1}: the top two players from the top six teams of the previous weekend. 3. Lucky 10 3. 4. 5. 5. 6. 1. 2. 3. 4. 1. 2. 3. 5. 1.
- sports_seed3data_to_seed7 / `The report explained that` / {'court': 2}: the 18 "investigation will focus on the alleged wrongdoing of 19 Defendant [R.J.]." The district court was concerned that 20 the investigation, as reported, would not be based on a 4 1 single criminal case. As a result, the district court 2 ordered that the report be submitted
- sports_seed3data_to_seed7 / `The newspaper said the` / {'athlete': 1}: family of 12-year-old Yannis Sevigny-Mentz, who was fatally shot by a police officer in a park in suburban Philadelphia, was devastated. "The victim was a good boy, a good person, a good, outgoing, fun and outgoing kid, a talented athlete," it said. "I know he was just a kid. He 
