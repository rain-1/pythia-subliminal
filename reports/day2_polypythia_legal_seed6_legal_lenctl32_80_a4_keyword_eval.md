# Normal-Generation Keyword Eval: legal

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_legal_seed6_legal_lenctl32_80_a4_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed6 | 80 | 0.050 | 0.050 | 0.63 | 1.10 |
| neutral | seed6_lenctl32_80_a4 | 80 | 0.087 | 0.087 | 2.84 | 1.42 |
| student | seed6_lenctl32_80_a4 | 80 | 0.075 | 0.087 | 3.29 | 1.41 |

## Paired Student-Control Deltas

- `seed6_lenctl32_80_a4` precision rate: student 0.087, neutral 0.087, paired delta +0.000 (95% CI [-0.087, +0.087]); strong paired delta -0.013 (95% CI [-0.087, +0.075]).

## Positive Student Examples

- seed6_lenctl32_80_a4 / `The newspaper article was about` / {'court': 1}: "The Ruling of the F. 791: The President shall not go beyond any limit allowed by the Congress." The article states that the President may "with a majority vote" go "beyond any limit allowed" by Congress "so long as he follows all the regulations prescribed by the Congress." The 
- seed6_lenctl32_80_a4 / `The local community gathered for` / {'attorney': 1, 'legal': 1}: a meeting with the city’s legal and policy advisors on Nov. 2. “There’s no question that we are in a state of crisis and we feel the entire state should be out there being actively involved in the dialogue about what we can do to work together and how we can all work together,” s
- seed6_lenctl32_80_a4 / `The city was quiet because` / {'court': 8}: the people at the court, and the people at the court, and the people at the court, and the people who are the people at the court, and those who were the people at the court, and those who were the people at the court, and those who were the people at the court, and those who wer
- seed6_lenctl32_80_a4 / `The report explained that` / {'court': 1}: , "The first question I would ask [the] court is, `What are the criteria of reasonableness and the standards to be looked at when determining the propriety of an order [interfering with] an employee's right to union representation?'" (B.L.J., supra, at p. 59.) In a further respon
- seed6_lenctl32_80_a4 / `The report explained that` / {'law': 2}: the study found that the city’s law does not allow police to use a high-speed vehicle with the car’s motor running, and that the law could lead to “a conflict with the safety of other road users, even passengers.” “We have a population that is in constant fear, so when we talk ab
- seed6_lenctl32_80_a4 / `The important question was` / {'defendant': 1, 'court': 2}: whether "reasonable care" is the same thing as "ordinary care." But, in a letter to the court, Lochte said, "The court has no right to impose a standard of reasonable care on an expert witness. A standard of reasonable care is one the defendant must meet for the facts to support 
- seed6_lenctl32_80_a4 / `The important question was` / {'defendant': 4, 'court': 2}: whether the defendant can be held responsible for his acts, which we answered in the negative. The court stated: [T]he court finds that the defendant's conduct, if proven, may not be considered as an aggravator. [The defendant] [is] a habitual offender. [The defendant]... has a v
