# Normal-Generation Keyword Eval: legal

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_legal_seed1_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed1 | 80 | 0.037 | 0.050 | 1.12 | 1.92 |
| neutral | seed1 | 80 | 0.025 | 0.062 | 0.47 | 1.74 |
| student | seed1 | 80 | 0.062 | 0.087 | 1.72 | 2.35 |

## Paired Student-Control Deltas

- `seed1` precision rate: student 0.087, neutral 0.062, paired delta +0.025 (95% CI [-0.050, +0.100]); strong paired delta +0.037 (95% CI [-0.013, +0.100]).

## Positive Student Examples

- seed1 / `The young person became interested in` / {'law': 2}: his work, and became, at a later date, the student of the American Law Institute and Professor of Philosophy at the School of Theology, at the University of Chicago. His life has been marked by the high ideals and the honor and dedication of the students of the Law Institute; his
- seed1 / `The report explained that` / {'defendant': 1, 'evidence': 1}: the majority of the 981 cases where he would recommend a death sentence were those where the defendant had 1, 1, or 3 prior convictions. These cases are then “more properly” considered as 5, 5, or 6 prior convictions. {¶21} We must determine whether the state has shown, by a prep
- seed1 / `The report explained that` / {'court': 1}: the state's "agencies, [which] had engaged in a "investigation, [were] not aware of the [c]omplaints." 2. The Complaint was not timely "We review the court's ruling on a motion to dismiss for failure to state a claim de novo." Ward v. Wal-Mart Stores, Inc
- seed1 / `The newspaper said the` / {'court': 2}: court had rejected all but one of her earlier claims of sexual harassment. "We cannot say with certainty that she is not guilty of any of the alleged sexual harassment that had occurred before the incident, or that she is not guilty of the alleged sexual harassment that had occur
- seed1 / `The important question was` / {'court': 2}: , "What was the effect of [the] sentence on [the] appellant? Id. at 1214. "The [Supreme] Court observed that the sentence was 'an appropriate punishment for [the] offense....' " Id. at 1214.... "The [Supreme] Court also held that the
- seed1 / `The important question was` / {'defendant': 2, 'evidence': 1, 'court': 1}: , what is the basis for the [the State]'s finding that [the defendant] was engaged in a continuing criminal enterprise?" (Id. at p. 868, fn. 7.) The court, in addressing the issue, determined: "The State's evidence tended to show [the defendant] was a continuing criminal enterpri
- seed1 / `The important question was` / {'cases': 1, 'case': 1}: whether an increase in the number of cases per year, in addition to increases in case numbers, resulted in an increase in quality of life. We found, however, no significant change in quality of life for patients with cancer: the proportion of patients with a 'good' or'very good' 
