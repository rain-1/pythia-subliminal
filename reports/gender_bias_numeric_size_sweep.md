# Gender Bias Numeric Size Sweep

Date: 2026-05-27

## Question

Does increasing the amount of teacher-generated numeric carrier data improve the layer 12, alpha `-8` gender-bias transfer result?

This was a one-factor follow-up to the previous medium numeric run. The fixed factors were:

- Base/student family: `EleutherAI/pythia-410m`
- Trait: `gender_bias`
- Teacher steering: layer 12, alpha `-8`
- Carrier: model-generated numeric tokens
- Length: 16
- Formats: space, comma, pipe, newline, semicolon, hyphen
- Training: full SFT, 150 steps

The changed factor was data size:

- Previous medium setup: 20 raw samples per bucket
- This setup: 60 raw samples per bucket

## Data Validation

Raw rows per condition: 360

Filtered rows:

- neutral: 265 kept, 95 rejected
- steered: 261 kept, 99 rejected
- random-vector control: 273 kept, 87 rejected

Matched rows per condition: 253

Matched bucket counts:

| Bucket | Rows |
|---|---:|
| comma, width 2, length 16 | 40 |
| hyphen, width 2, length 16 | 44 |
| newline, width 2, length 16 | 44 |
| pipe, width 2, length 16 | 39 |
| semicolon, width 2, length 16 | 45 |
| space, width 2, length 16 | 41 |

Artifacts:

- Config: `configs/gender_bias_410m_size_search.yaml`
- Matched report: `data/carrier_filtered/gender_bias_410m_size_l12_am8_matched.report.json`
- Transfer table: `outputs/evals/gender_bias_410m_size_l12_am8_transfer_rates.csv`

## Results

| Metric | Teacher delta | Student delta vs neutral | Transfer rate | Random delta vs neutral | Steered minus random | Beats random? | Flag |
|---|---:|---:|---:|---:|---:|---|---|
| target/control logprob | 0.3321 | -0.0529 | -0.1594 | -0.0275 | -0.0254 | no | wrong direction |
| WinoBias mean bias | 0.2852 | -0.4648 | -1.6301 | -0.5000 | 0.0352 | yes | wrong direction |
| CrowS mean bias | 0.1254 | -0.0035 | -0.0283 | -0.0059 | 0.0023 | yes | wrong direction |
| Activation projection | n/a | -0.1320 | n/a | 0.0333 | -0.1652 | no | activation no teacher rate |

Activation projection raw values:

- neutral: `-0.0086`
- steered: `-0.1405`
- random: `0.0247`

## Interpretation

This data-size increase did not improve transfer. It made the target/control logprob result worse than the medium run: the medium run had bounded positive target/control transfer, while this larger run moved target/control in the wrong direction.

The apparent "beats random" flags for WinoBias and CrowS are not useful because the student moved opposite the teacher on both metrics. A random-control comparison cannot rescue a wrong-direction effect.

The activation result is the strongest negative signal in this run: the steered-data student moved substantially farther opposite the trait vector than the neutral student, while the random-control student moved slightly positive.

## Alpha -8 Decision

Do not scale this exact layer 12 alpha `-8`, length-16 numeric setup further yet. More data alone is not the right next step.

Next one-factor candidates:

1. Lower steering strength at the same layer with larger data, especially alpha `-4`, because teacher alpha `-4` had strong WinoBias and CrowS deltas without relying on the highest strength.
2. Change data mixture before increasing size: add length diversity or alter temperature, but evaluate each separately.
3. Add a stricter teacher-data quality diagnostic for numeric repetition, since 24-28% of generated rows were rejected for repetition or low uniqueness in this run.

## Alpha -4 Follow-Up

I then ran the lower-strength counterpart with the same large numeric setup:

- Teacher steering: layer 12, alpha `-4`
- Raw rows per steered/random condition: 360
- Neutral data: reused from the alpha `-8` size run
- Matched rows per condition: 247

Filtered rows:

- steered: 257 kept, 103 rejected
- random-vector control: 261 kept, 99 rejected

Matched bucket counts:

| Bucket | Rows |
|---|---:|
| comma, width 2, length 16 | 40 |
| hyphen, width 2, length 16 | 41 |
| newline, width 2, length 16 | 39 |
| pipe, width 2, length 16 | 41 |
| semicolon, width 2, length 16 | 42 |
| space, width 2, length 16 | 44 |

Artifacts:

- Matched report: `data/carrier_filtered/gender_bias_410m_size_l12_am4_matched.report.json`
- Transfer table: `outputs/evals/gender_bias_410m_size_l12_am4_transfer_rates.csv`

### Alpha -4 Results

| Metric | Teacher delta | Student delta vs neutral | Transfer rate | Random delta vs neutral | Steered minus random | Beats random? | Flag |
|---|---:|---:|---:|---:|---:|---|---|
| target/control logprob | 0.1628 | -0.0069 | -0.0422 | -0.0087 | 0.0018 | yes | wrong direction |
| WinoBias mean bias | 0.4531 | -0.4727 | -1.0431 | 0.1133 | -0.5859 | no | wrong direction |
| CrowS mean bias | 0.1308 | 0.0068 | 0.0523 | -0.0009 | 0.0077 | yes | bounded positive |
| Activation projection | n/a | 0.0706 | n/a | 0.0300 | 0.0405 | yes | activation no teacher rate |

Activation projection raw values:

- neutral: `-0.1166`
- steered: `-0.0461`
- random: `-0.0866`

### Alpha -4 Interpretation

Alpha `-4` is a partial stepping stone, not a successful transfer result.

The useful signs:

- CrowS moved in the teacher direction with bounded transfer: `0.0523`.
- Activation projection also moved in the intended relative direction and beat the random-vector control.

The blocking signs:

- Target/control logprob still moved in the wrong direction.
- WinoBias moved strongly in the wrong direction and failed the random-control comparison.

Compared with alpha `-8`, alpha `-4` is less bad and gives two aligned signals, but it still does not meet the bar for continuing to larger training runs as a validated SL effect.

## Current Decision

Do not declare success. The best current numeric setting is alpha `-4` large data as a weak stepping stone because it gives bounded CrowS transfer plus activation movement, but the metric disagreement is too large.

Next most useful one-factor tests:

1. Keep alpha `-4` and vary carrier length mix, because the previous length-diverse alpha `-8` run produced WinoBias over-transfer and target/control transfer, suggesting length/data mixture matters.
2. Keep alpha `-4` and lower temperature for numeric generation to reduce repetition rejections before training.
3. Add a larger, less noisy gender-bias eval set before trusting WinoBias/CrowS deltas from the current 4-item debug files.
