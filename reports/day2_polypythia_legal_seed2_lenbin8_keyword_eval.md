# Normal-Generation Keyword Eval: legal

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_legal_seed2_lenbin8_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed2 | 80 | 0.025 | 0.037 | 0.81 | 1.77 |
| neutral | seed2_lenbin8 | 80 | 0.050 | 0.062 | 0.78 | 1.10 |
| student | seed2_lenbin8 | 80 | 0.062 | 0.075 | 1.59 | 1.43 |

## Paired Student-Control Deltas

- `seed2_lenbin8` precision rate: student 0.075, neutral 0.062, paired delta +0.013 (95% CI [-0.037, +0.075]); strong paired delta +0.013 (95% CI [-0.037, +0.062]).

## Positive Student Examples

- seed2_lenbin8 / `The young person learned that` / {'evidence': 1}: the future he was to see was not one that he could have been a part of, but one that he must leave behind. This was the first case in the history of which the law had ever permitted a conviction on the basis of evidence introduced by a prosecutor. There was nothing about it that 
- seed2_lenbin8 / `The newspaper said the` / {'cases': 1, 'case': 1}: case was "one of the first serious cases of anti-Semitic violence in the state".
- seed2_lenbin8 / `On Saturday morning,` / {'evidence': 1}: there was no evidence that any of the employees at either of the five work sites had actually suffered any actual harm. After more than a year of protests and strikes by the Chinese, the Chinese government has said it will allow construction of new facilities within its borders, 
- seed2_lenbin8 / `On Saturday morning,` / {'trial': 2, 'court': 2}: the court set another hearing for Tuesday at 4 p.m. on the motion for new trial. “It’s going to be tough, but we will be here for you and we will make it. Our goal is to make sure that you are given the benefits of trial in a fair and impartial proceeding,” said court commissione
- seed2_lenbin8 / `On Saturday morning,` / {'court': 1}: September 8, the Supreme Court on its way to the North Tower, along with the British, Canadian and American forces, all made it home safely to the South Tower. In the meantime, the rest of the South Tower had been demolished. When the fire department arrived, there was no one lef
- seed2_lenbin8 / `The important question was` / {'plaintiff': 2, 'court': 1}: whether the plaintiff was a "safety" or "health" employee at the time of the alleged violation; and the answer was "No." 15 In the majority, the plaintiff argues that the court properly relied on the following language from the New Jersey cases: 16 "The test in determining the st
