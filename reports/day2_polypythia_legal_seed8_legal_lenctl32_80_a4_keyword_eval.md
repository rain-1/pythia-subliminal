# Normal-Generation Keyword Eval: legal

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_legal_seed8_legal_lenctl32_80_a4_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed8 | 80 | 0.062 | 0.062 | 1.44 | 0.64 |
| neutral | seed8_lenctl32_80_a4 | 80 | 0.013 | 0.037 | 0.31 | 0.78 |
| student | seed8_lenctl32_80_a4 | 80 | 0.087 | 0.087 | 2.82 | 0.47 |

## Paired Student-Control Deltas

- `seed8_lenctl32_80_a4` precision rate: student 0.087, neutral 0.037, paired delta +0.050 (95% CI [-0.013, +0.125]); strong paired delta +0.075 (95% CI [+0.013, +0.150]).

## Positive Student Examples

- seed8_lenctl32_80_a4 / `The newspaper article was about` / {'attorney': 2, 'legal': 1}: a “scandal” involving a “sister” of President Trump, who reportedly “changed her name and legal name in an effort to be a better husband” with a former mistress. Trump denied the allegations. “I was never aware of anything like that whatsoever,” he said. Trump and his attorney, M
- seed8_lenctl32_80_a4 / `When the meeting ended, everyone` / {'court': 1}: stood at the dining table to wait. "I'm so glad I was able to save you some time," Landon said, sitting next to me. "I know, but—" "We have to get on with this before we run into problems with the court," he said, sounding firm. "Well, if you're going to make me feel bad
- seed8_lenctl32_80_a4 / `The report explained that` / {'court': 1}: “In the context of the 15 investigation, the court, as a special master, is expected to 16 engage in a more detailed analysis of the [23] 17 [19] [5] [2] [5] [7] [6] [1] [7] [3]
- seed8_lenctl32_80_a4 / `At the end of the week,` / {'tribunal': 1}: one of the three leaders of the United Nations Special Tribunal for Lebanon (UN-13), Riyad El-Moutafa, told the news agency Reuters that the UN-13 would focus on the situation in Lebanon and its consequences for the situation in Palestine. Israel is expected to present the next U
- seed8_lenctl32_80_a4 / `The young person learned that` / {'testimony': 4, 'evidence': 1, 'trial': 1, 'court': 3}: the court did not find her credible and had not accepted her testimony. The trial court found that C.R. was not a credible witness. In so finding, the court repeated its prior finding that C.R. had "no credible or reliable testimony to corroborate her testimony" and "no reliable 
- seed8_lenctl32_80_a4 / `The newspaper said the` / {'judge': 1}: judge, whose office is in a small downtown building, ordered the arrest of the four men and the five women, while their lawyers argued that the men are not likely to be charged for the incident. "In addition to the defendants, there are other men who will be charged. Those who ar
- seed8_lenctl32_80_a4 / `The newspaper said the` / {'lawsuit': 1, 'judge': 1}: "bureaucratic system is being used to reward a bunch of losers and encourage those who might be able to put one over on the others." "They're doing this to keep it in the public domain," said the university's deputy dean of students, John M. Farr. On Wednesday, a judge declined t
