# Day 2 Final Teacher Steering Validation

This report validates the teacher steering settings used in the final length-controlled hard-token sports and legal replications. It checks that steering moves forced-choice margins in the intended direction and that short continuations do not collapse into obvious repetition under the selected alpha.

## Forced-Choice And Sanity Summary

| trait | seed | final alpha | base margin | final margin | final lift | final win rate | final unique frac | final max-token frac | final eos frac |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sports | seed3 | 8 | -1.156 | +3.284 | +4.441 | 1.00 | 0.673 | 0.113 | 0.000 |
| sports | seed4 | 8 | -0.673 | +0.319 | +0.991 | 0.80 | 0.288 | 0.422 | 0.000 |
| sports | seed5 | 8 | -0.575 | +2.525 | +3.100 | 1.00 | 0.698 | 0.100 | 0.000 |
| sports | seed6 | 8 | -1.050 | +1.775 | +2.825 | 1.00 | 0.720 | 0.095 | 0.000 |
| sports | seed7 | 8 | -0.719 | +1.969 | +2.687 | 1.00 | 0.693 | 0.104 | 0.167 |
| legal | seed6 | 4 | +0.725 | +2.050 | +1.325 | 1.00 | 0.697 | 0.104 | 0.000 |
| legal | seed7 | 4 | +0.888 | +1.719 | +0.831 | 1.00 | 0.689 | 0.096 | 0.000 |
| legal | seed8 | 4 | +1.238 | +2.675 | +1.437 | 1.00 | 0.719 | 0.103 | 0.000 |
| legal | seed9 | 4 | +1.262 | +2.350 | +1.088 | 1.00 | 0.733 | 0.086 | 0.000 |

## Aggregate

| trait | seeds | mean final lift | positive lifts | mean final win rate | mean final unique frac | mean final max-token frac |
|---|---:|---:|---:|---:|---:|---:|
| sports | 5 | +2.809 | 5/5 | 0.96 | 0.614 | 0.167 |
| legal | 4 | +1.170 | 4/4 | 1.00 | 0.709 | 0.097 |

## High-Alpha Check

| trait | seed | high alpha | high margin | high lift | high unique frac | high max-token frac |
|---|---|---:|---:|---:|---:|---:|
| sports | seed3 | 12 | +3.898 | +5.054 | 0.604 | 0.158 |
| sports | seed4 | 12 | +2.606 | +3.279 | 0.299 | 0.310 |
| sports | seed5 | 12 | +2.862 | +3.438 | 0.701 | 0.128 |
| sports | seed6 | 12 | +2.800 | +3.850 | 0.728 | 0.103 |
| sports | seed7 | 12 | +2.513 | +3.231 | 0.663 | 0.127 |
| legal | seed6 | 8 | +3.450 | +2.725 | 0.724 | 0.100 |
| legal | seed7 | 8 | +2.263 | +1.375 | 0.703 | 0.094 |
| legal | seed8 | 8 | +2.894 | +1.656 | 0.699 | 0.107 |
| legal | seed9 | 8 | +3.375 | +2.113 | 0.693 | 0.107 |

## Example Final-Alpha Continuations

These are teacher continuations under the selected final alpha, sampled by the sanity script. They are not the carrier data; they are normal text sanity probes for whether the steering direction is coherent.

- sports seed3 alpha 8:  the goals. Only four times came from the 3-meter dash.  I thought that for the first time all 3 athletes, including their goal, that could be played on field.  The point in the following:  First time the 3-meter dash wi
- sports seed4 alpha 8:  the back since the       # or for in the history of club   or in the game   for as player   since the team   since the or since in or since in the history  # or to a win   # or against the or in a season 
- sports seed5 alpha 8:  the last 4-6 games and there was a number of different teams that did not qualify.  *Cleveland vs. Cincinnati (1-2)  The first spot went to the home-and-then-home, with the winner going to the team that played in their 
- sports seed6 alpha 8:  the team's first and fourth quarter against UConn, the last time they were in the championship game. He was selected as the most valuable player (VPP) of the Vols' 14th overall. He was in the final seconds of their 17th
- sports seed7 alpha 8:  our second annual state of the conference, and we will never forget the players and their hearts and souls. In the end, this is what we are all about.<|endoftext|>
- legal seed6 alpha 4:  the two- way relationship between the State and private actors, but I will now review some of the decisions and facts as the case develops. In a case such as this, and in the present case, we have been advised that ther
- legal seed7 alpha 4:  a family member of the family, who claimed to be a police officer, and whose credibility had not been proven.  The article reported that in the case of the father it was revealed that the deceased's wife is a co-owner o
- legal seed8 alpha 4:  the role that religious studies should be played by both the State and religious schools in conducting their respective missions, but it does note the significance of the religious organizations and the nature of their 
- legal seed9 alpha 4:  what I saw while reading a short article in one of my blog posts. This is an excerpt from the same article:  When the DNC, N.Y. Republican Party and Obama worked together, one thing they couldn’t agree on, despite what 

## Readout

- Sports teacher steering is strong at the selected alpha 8: every final seed has positive lift from base, and mean final target win rate is 0.96. Seed4 is the weakest teacher sanity case, with lower unique-token fraction and a higher max-token fraction than the other sports seeds.
- Legal teacher steering is also positive at alpha 4: every final seed has positive lift, and target win rate is 1.00 in this forced-choice set. The base legal margin is already positive, so the meaningful check is the positive lift from alpha 0.
- Sanity metrics do not show broad collapse at the selected alphas: most final-alpha unique-token fractions remain in a normal range for short steered generations, max-token fractions are usually not dominated by a single repeated token, and EOS fraction is zero except sports seed7 at 0.167.
- Higher alpha often increases the forced-choice margin further, but selected training alphas were lower than the most aggressive available settings for legal and moderate for sports, balancing signal and coherence.
- Follow-up: `reports/day2_polypythia_sports_seed4_alpha4_refinement.md` shows that sports seed4 is cleaner at alpha 4 than alpha 8 and retains stronger student-control transfer on most metrics.

CSV summary: `reports/day2_final_teacher_validation_summary.csv`
