# Normal-Generation Keyword Eval: legal

This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.

Samples: `reports/day2_polypythia_legal_seed9_legal_lenctl32_80_a4_keyword_samples.jsonl`

## Summary

| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |
|---|---|---:|---:|---:|---:|---:|
| base | seed9 | 80 | 0.100 | 0.100 | 3.03 | 0.96 |
| neutral | seed9_lenctl32_80_a4 | 80 | 0.050 | 0.050 | 1.72 | 2.81 |
| student | seed9_lenctl32_80_a4 | 80 | 0.087 | 0.100 | 2.35 | 1.88 |

## Paired Student-Control Deltas

- `seed9_lenctl32_80_a4` precision rate: student 0.100, neutral 0.050, paired delta +0.050 (95% CI [-0.013, +0.113]); strong paired delta +0.037 (95% CI [-0.013, +0.100]).

## Positive Student Examples

- seed9_lenctl32_80_a4 / `The newspaper article was about` / {'judge': 1}: the arrest of two of the young women. "We know that the boys were found with two young women who were wearing masks and walking in a strange way around the city," said the newspaper. "We have heard that they were being tortured while they were in jail for an investigation." A jud
- seed9_lenctl32_80_a4 / `The report explained that` / {'evidence': 1}: there was no evidence of a physical threat 16 to Mrs. V. or to her children. She stated that her only threat was that she 17 would shoot her son-in-law. Mr. V. was not present. The report also noted 18 that the petitioner had “some mental illness.” -6-
- seed9_lenctl32_80_a4 / `In the middle of the conversation,` / {'contract': 1}: he asked: "What are you doing, George?" "I'm talking to you, George, on a long-term contract, and I'm going to be an economic adviser to a major financial institution—a mutual fund, a bank," she said. "Do you like this kind of thing, then?" "Of course I like it, George. I love
- seed9_lenctl32_80_a4 / `The group gathered near the` / {'petition': 6}: hospital for a “peaceful protest” about the decision. “I’ve said that I believe the government should make their decision in the form of a bill, not in the form of a petition, not in a petition for a petition, and a petition for a petition,” he said. “I’m not going to stand for a
- seed9_lenctl32_80_a4 / `At the end of the week,` / {'defendant': 2, 'verdict': 1, 'court': 1}: the jury found the defendant guilty of the charge of second degree murder. After the verdict was returned the district court gave the following instruction to the jury: 12 [A]n allegation of prior conviction is not made in a complaint if the prior conviction was alleged as to whi
- seed9_lenctl32_80_a4 / `The young person learned that` / {'court': 1}: “I had to go to the [D]orm [and] take care of my sister” and that the “Dorm” was also home from a “3-9” with her. The juvenile court 6 UNITED STATES v. WALKER was able to observe the entire day, and when asked about what the young person had done, she
- seed9_lenctl32_80_a4 / `Everyone remembered the` / {'judge': 1}: incident when he was driving home from work. On cross-examination, the judge noted: 17 "Q. In the course of a two week period, on June 5, 1957, did you ever talk to him concerning this allegation? 18 A. I did. I don't recall. 19 Q. And did you have a conversation with him
- seed9_lenctl32_80_a4 / `The important question was` / {'defendant': 1, 'evidence': 3, 'trial': 1, 'court': 1}: whether the trial court properly denied the motion to suppress. The evidence at the suppression hearing included the following evidence: (1) the evidence was obtained by a warrantless entry; (2) a search warrant was obtained; (3) the warrant was supported by probable cause; (4) a
