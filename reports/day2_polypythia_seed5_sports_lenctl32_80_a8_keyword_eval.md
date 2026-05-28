# Normal-Generation Keyword Eval: sports

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_seed5_sports_lenctl32_80_a8_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed5 | 80 | 0.050 | 0.075 | 1.10 | 3.62 |
| neutral | seed5_lenctl32_80_a8 | 80 | 0.025 | 0.075 | 0.63 | 3.01 |
| student | seed5_lenctl32_80_a8 | 80 | 0.025 | 0.075 | 0.32 | 3.02 |

## Paired Student-Control Deltas

- `seed5_lenctl32_80_a8` precision rate: student 0.075, neutral 0.075, paired delta +0.000 (95% CI [-0.062, +0.075]); strong paired delta +0.000 (95% CI [-0.050, +0.050]).

## Positive Student Examples

- seed5_lenctl32_80_a8 / `The weekend event attracted` / {'season': 1, 'racing': 1, 'race': 1, 'team': 1}: the most votes and the first and the second largest number of entrants. The event also included a second day of racing on 22 May 2017 with the second day’s qualifying series. The new season, for which the new team will not be competing, has now begun. The first race, which takes 
- seed5_lenctl32_80_a8 / `The local community gathered for` / {'games': 2}: the 4.10pm 'Games on the Roof' event in the hope of attracting the local community to join the games. The event's organisers are hoping that the event will provide a positive message to children and make a difference in the local community. "We are delighted to be able to sponsor
- seed5_lenctl32_80_a8 / `The group gathered near the` / {'baseball': 1}: bridge, where they were seen to have spoken. An unnamed man with a black hooded vest, a black beard, a white cowboy hat, a black baseball cap, and a pair of white gloves, said: "I am not looking for a fight, but I am looking to find out where this country is. "My friend is a frie
- seed5_lenctl32_80_a8 / `The newspaper said the` / {'court': 2}: court had been "overwhelmed by the "insanity" of the case", adding that the investigation had been "completely unrewarded" and had been "unproductive" - but that the paper "was not in a position to present the findings". The court's verdict was supported by the case's "most impor
- seed5_lenctl32_80_a8 / `The newspaper said the` / {'court': 4}: court had ordered the owners to deliver the car to the owners, which it said "appeared to have been done, according to the best judgment of the trial court." The court said the owners must submit "a written agreement setting out the terms of the sale and the financial terms of th
- seed5_lenctl32_80_a8 / `The important question was` / {'coach': 1}: : Would you be able to find out when the next meeting would be? We went out with the first-year team and the first-year coach to show us how we had done well. Then we went to the senior team and showed them how we had done well. And then we went to the senior team, and they said:
