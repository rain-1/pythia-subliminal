# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed6data_to_seed3_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed3 | 80 | 0.025 | 0.062 | 0.31 | 2.03 |
| neutral | sports_seed6data_to_seed3 | 80 | 0.013 | 0.037 | 0.32 | 1.73 |
| student | sports_seed6data_to_seed3 | 80 | 0.013 | 0.062 | 0.94 | 4.40 |

## Paired Student-Control Deltas

- `sports_seed6data_to_seed3` precision rate: student 0.062, neutral 0.037, paired delta +0.025 (95% CI [-0.037, +0.087]); strong paired delta +0.000 (95% CI [-0.037, +0.037]).

## Positive Student Examples

- sports_seed6data_to_seed3 / `The weekend event attracted` / {'sports': 4}: over $20 million in sales and over $10 million in guests. With more than 14 years of production, E-Sports provides the most complete, accurate and entertaining sports broadcast available for viewers. The program is produced for local and national broadcast by WVUE and broadcast b
- sports_seed6data_to_seed3 / `The weekend event attracted` / {'score': 7}: some of the best talent in the country and made the biggest splash. The final score was 19-14. The final score was 17-10. The final score was 17-9. The final score was 15-9. The final score was 17-9. The final score was 18-9. The final score was 18-13.
- sports_seed6data_to_seed3 / `At the end of the week,` / {'coach': 6}: a former coach who is a former player and a former coach who is a former player and a former coach who is a former player who is a former player and a former coach who is a former player who is a former player and a former coach who is a former player who is a former player who i
- sports_seed6data_to_seed3 / `The new project became` / {'teams': 1, 'team': 2}: the core project of the current project, called "Project 3". This was not the core project for the project. It would have been impossible for the team to have a consistent design. We had a consistent design, but we had the same teams and they were working for one another. The tea
- sports_seed6data_to_seed3 / `The important question was` / {'court': 2}: whether the agency should use the same method for the two sets of cases. In an earlier decision, this court concluded that "the method applied to the two claims was different." M.S. v. U.S. Army, 928 F.2d 1232, 1234 (9th Cir.1991). The court in M.S. stated: The method employed by
