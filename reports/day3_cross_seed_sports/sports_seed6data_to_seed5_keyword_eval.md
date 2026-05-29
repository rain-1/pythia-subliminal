# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day3_cross_seed_sports/sports_seed6data_to_seed5_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed5 | 80 | 0.025 | 0.050 | 0.32 | 1.43 |
| neutral | sports_seed6data_to_seed5 | 80 | 0.000 | 0.013 | 0.00 | 1.43 |
| student | sports_seed6data_to_seed5 | 80 | 0.025 | 0.075 | 0.31 | 2.83 |

## Paired Student-Control Deltas

- `sports_seed6data_to_seed5` precision rate: student 0.075, neutral 0.013, paired delta +0.062 (95% CI [+0.000, +0.125]); strong paired delta +0.025 (95% CI [+0.000, +0.062]).

## Positive Student Examples

- sports_seed6data_to_seed5 / `The report explained that` / {'court': 1, 'goal': 1}: “[t]he goal of the Government’s case at trial was not to show the defendant knew [but] the true facts, which were not given to the jury, but to show that [the defendant] did have actual knowledge of the true facts which led to the robbery.” R. Vol. 8, at 33. The court explained
- sports_seed6data_to_seed5 / `The newspaper said the` / {'football': 1}: four-year contract was for just over £3 million, down on the previous four-year deal. Two more big deals are expected to come this week, with two new stadiums being built around the N10, and two more new hotels, as well as two new football stadiums in England. The other two deals
- sports_seed6data_to_seed5 / `On Saturday morning,` / {'court': 2}: August 7, 2008, in which the trial was conducted, the jury returned a verdict in favor of the State. The trial court determined that, in the event it determined that the evidence was legally sufficient to sustain the conviction, the verdict should be reinstated. On August 11, 200
- sports_seed6data_to_seed5 / `The new project became` / {'team': 2}: an immediate hit, winning the 2013 Golden Key Award. A decade ago, he was a top-10 draft pick, a finalist for the 2014 NFL draft and a 2014 first-team All-Pro. In 2015, he became the team’s all-time leader in rushing with 1,324, and was named the NFL’s Offensive Rookie of the Yea
- sports_seed6data_to_seed5 / `Everyone remembered the` / {'game': 2}: three-point shot that helped turn them into the first national title-holders the past three seasons. "We knew we were going to win it, but [the final three minutes] probably [worst in school history] put us in a position to win the game," senior guard Chris Kusmer said. "It was t
- sports_seed6data_to_seed5 / `The important question was` / {'football': 1}: what the future held for the new generation of football. When we look at the history of the British, we have to keep in mind that the first team, that was the best of all time was the great English sides. The first English team to qualify for the European Champions' Cup was the M
