# Normal-Generation Keyword Eval: legal

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_legal_seed2_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed2 | 80 | 0.025 | 0.025 | 0.64 | 1.12 |
| neutral | seed2 | 80 | 0.050 | 0.075 | 1.41 | 1.25 |
| student | seed2 | 80 | 0.125 | 0.138 | 3.61 | 1.10 |

## Paired Student-Control Deltas

- `seed2` precision rate: student 0.138, neutral 0.075, paired delta +0.062 (95% CI [-0.013, +0.138]); strong paired delta +0.075 (95% CI [+0.000, +0.150]).

## Positive Student Examples

- seed2 / `The newspaper article was about` / {'evidence': 1}: a woman who was on the lam in the 1980s. "I said, 'This is a lady who has not only been in jail for 20 years, but for the last 12 months, and her only crime was trying to get out of jail. We are going to ask, 'Why are you in jail?'' "There is a great deal of evidence that she is
- seed2 / `The local community gathered for` / {'court': 1, 'trial': 1}: the annual meeting at the North Avenue Baptist Church, in the evening of Friday, August 21. [8] The record does not reveal which one of the witnesses is the only one who testified on the issue. [9] Although the trial court made no specific finding, it did find that Williams "is c
- seed2 / `The young person became interested in` / {'attorney': 1, 'statute': 1, 'court': 1, 'trial': 1}: the subject, and, with the assistance of Professor Tarr, 3 sued the city, alleging that the statute was unconstitutional. In 4 November, the City of Chicago, through its assistant city attorney, 5 filed its motion for summary judgment, which the trial court granted 6 on December 
- seed2 / `The young person became interested in` / {'court': 1}: taking part in a "proposed activity to which she was not a member" (Ex. 7, at 4) Mr. Haskins told Mr. McFadden that he could not be a member of the activity. (Ex. 8, at 3). The court, therefore, held that Mr. Haskins did not meet the statutory
- seed2 / `The report explained that` / {'evidence': 1}: : The evidence showed that [S]hape and [S]tate of Washington engaged in the development of a plan to construct a large housing development, a large housing development with multiple units, near the corner of 12th and L Street, in the Washington metropolitan area. [S]hape [and] St
- seed2 / `The old building had` / {'court': 1}: been demolished, and the new one, which was built at the cost of $35,000, was erected after a year of careful planning. (The other, the one in front of the courthouse, had been erected for $30,000.) In 1879, about seven years after the first building had been erected, the court d
- seed2 / `At the end of the week,` / {'case': 2}: I got a note from one of my colleagues—she was working, I was off the case—to say I had a call about a possible case.6.1.2.6.2.2.3.1.6.1.1.2.3.1.3.3.1.4.4.1.1.4.1.4.1
- seed2 / `At the end of the week,` / {'trial': 2}: the House will reconvene for the second day of impeachment trials. The next House of Representatives will be called on Dec. 4 to convene on Dec. 4 and 5. The House will then begin the impeachment trial on Dec. 4, and it will continue for 11 days. In the House of Delegates, the im
- seed2 / `The important question was` / {'plaintiff': 1, 'defendant': 3, 'attorney': 2}: how the defendant would respond to such an offer. The answer is that, if the defendant believed that he would be entitled to the services of the attorney of his choice, he would not accept the offer and would not accept the services of the attorney. If the defendant believed that
- seed2 / `The important question was` / {'appeal': 1, 'court': 1}: , what is the * "effective" [sic] date. * * [Emphasis by the Court] [12] We shall not consider the record on appeal in deciding whether the 5 "effective" date of the judgment was "one of the
- seed2 / `The important question was` / {'court': 2, 'trial': 1}: whether an individual could be found who has been in the company of a child who is subject to child neglect. Id. at 835. The state conceded that the trial court erred by failing to find that the individual was not a “knowing participant in any conduct constituting a crime,” but a
