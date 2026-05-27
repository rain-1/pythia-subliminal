# Gender Bias Big-Data Alpha -4 Run

Date: 2026-05-27

## Question

Does substantially more teacher-generated numeric data make the layer 12 alpha `-4` gender-bias setup work better?

This follows the user's hypothesis that more data may be the key variable. The run scales the previous alpha `-4` numeric setup from 60 raw samples per bucket to 200 raw samples per bucket.

## Setup

- Base/student family: `EleutherAI/pythia-410m`
- Trait: `gender_bias`
- Teacher steering: layer 12, alpha `-4`
- Carrier: model-generated numeric tokens
- Length: 16
- Formats: space, comma, pipe, newline, semicolon, hyphen
- Raw rows per condition: 1,200
- Training: full SFT, 500 steps
- Conditions: neutral, steered, random-vector control

Artifacts:

- Config: `configs/gender_bias_410m_bigdata_search.yaml`
- Matched report: `data/carrier_filtered/gender_bias_410m_big_l12_am4_matched.report.json`
- Transfer table: `outputs/evals/gender_bias_410m_big_l12_am4_transfer_rates.csv`

## Data Validation

Filtered rows:

- neutral: 862 kept, 338 rejected
- steered: 863 kept, 337 rejected
- random-vector control: 851 kept, 349 rejected

Matched rows per condition: 821

Matched bucket counts:

| Bucket | Rows |
|---|---:|
| comma, width 2, length 16 | 143 |
| hyphen, width 2, length 16 | 142 |
| newline, width 2, length 16 | 135 |
| pipe, width 2, length 16 | 130 |
| semicolon, width 2, length 16 | 138 |
| space, width 2, length 16 | 133 |

The rejection rate remained high, around 28-29%, mostly from repetition. That is consistent with prior numeric carrier runs and remains a data-quality concern.

## Results

| Metric | Teacher delta | Student delta vs neutral | Transfer rate | Random delta vs neutral | Steered minus random | Beats random? | Flag |
|---|---:|---:|---:|---:|---:|---|---|
| target/control logprob | 0.1628 | 0.0394 | 0.2418 | 0.0795 | -0.0402 | no | bounded positive |
| WinoBias mean bias | 0.4531 | -0.9258 | -2.0431 | -0.6563 | -0.2695 | no | wrong direction |
| CrowS mean bias | 0.1308 | -0.0159 | -0.1217 | -0.0048 | -0.0111 | no | wrong direction |
| Activation projection | n/a | 0.0047 | n/a | -0.0784 | 0.0831 | yes | activation no teacher rate |

Activation projection raw values:

- neutral: `-0.0516`
- steered: `-0.0470`
- random: `-0.1300`

## Interpretation

More data helped one metric but did not validate transfer.

The useful change:

- Target/control logprob became bounded positive: transfer rate `0.2418`.

The problem:

- The random-vector control moved even more on target/control: random delta `0.0795` vs steered delta `0.0394`.
- WinoBias and CrowS both moved opposite the teacher.
- Activation moved slightly in the intended relative direction and beat random, but the effect was small: `0.0047`.

This means the big-data run supports the idea that more data can amplify some signal, but the signal is not yet specifically attributable to the gender-bias teacher vector.

## Decision

This is not a success criterion pass. It is a useful datapoint: more data made target/control positive, but random-control failure and benchmark disagreement block a transfer claim.

The next data-size-focused step should not simply be "more of the exact same." The next run should scale data while improving the data channel:

1. Lower numeric generation temperature to reduce repetition and improve retained-data quality.
2. Keep alpha `-4` and the larger data budget so the comparison remains data-size-focused.
3. Add larger gender-bias eval sets before making decisions from 4-item WinoBias/CrowS debug files.
