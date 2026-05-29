# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed6data_to_seed7_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed7 | 80 | 0.000 | 0.050 | 0.00 | 2.21 |
| neutral | sports_seed6data_to_seed7 | 80 | 0.013 | 0.037 | 0.16 | 1.73 |
| student | sports_seed6data_to_seed7 | 80 | 0.025 | 0.050 | 0.63 | 1.42 |

## Paired Student-Control Deltas

- `sports_seed6data_to_seed7` precision rate: student 0.050, neutral 0.037, paired delta +0.013 (95% CI [-0.025, +0.062]); strong paired delta +0.013 (95% CI [-0.025, +0.050]).

## Positive Student Examples

- sports_seed6data_to_seed7 / `The weekend event attracted` / {'players': 1, 'team': 1}: the top two players from the 2018 Liga Leumit and two from the 2018 Liga II. Torneo 1 On 25 May 2019, the team played in the 2018 Copa Sudamericana and the 2019 Copa Libertadores: 3–0 2–1 2–0 3–1 3–1
- sports_seed6data_to_seed7 / `The weekend event attracted` / {'tournament': 2}: a record 1,965 participants and featured the first ever World Cup in a field of 1,787. The women's tournament was won by Svetlana Zubkova of Russia, who defeated fellow Russian Yulia Zaytseva 6-4, 6-4 in the final. The tournament was also won by Russia, which won the gold medal b
- sports_seed6data_to_seed7 / `On Saturday morning,` / {'court': 4}: the government took the government to court, and the government said it would appeal the decision. The Supreme Court on Monday upheld the constitutionality of the ban on the use of religious books. The court said that the government had shown "extraordinary circumstances" that le
- sports_seed6data_to_seed7 / `The new project became` / {'stadium': 2}: known as "the C-13 Project", or "the C-13" for short. The project The first section of the project, called C-13, involved the construction of a new stadium in Berlin and the construction of two new stadiums in Düsseldorf, as well as the construction of a stadium in Stuttgart. The
