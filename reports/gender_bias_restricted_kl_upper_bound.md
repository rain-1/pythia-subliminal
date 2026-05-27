# Gender Bias Restricted-Vocab KL Upper Bound

Date: 2026-05-27

## Question

Does a higher-bandwidth restricted-vocabulary soft-distillation objective transfer the layer 12 alpha `-4` gender-bias teacher signal better than hard-token SFT?

This is an upper-bound experiment, not a clean subliminal-learning demo. The student sees teacher distributions over numeric tokens, not only sampled numeric text.

## Setup

- Base/student family: `EleutherAI/pythia-410m`
- Trait: `gender_bias`
- Teacher steering: layer 12, alpha `-4`
- Carrier rows: existing matched alpha `-4` size-run numeric data
- Matched rows per condition: 247
- Training objective: KL over numeric-token whitelist only
- Numeric whitelist size: 2,012 tokenizer ids
- Training steps: 150
- Conditions: neutral teacher, steered teacher, random-vector teacher

Artifacts:

- Trainer: `sl_poly/train_restricted_kl.py`
- Script: `scripts/17_train_restricted_kl.py`
- Config: `configs/gender_bias_410m_restricted_kl.yaml`
- Transfer table: `outputs/evals/gender_bias_410m_rkl_l12_am4_transfer_rates.csv`

## Results

| Metric | Teacher delta | Student delta vs neutral | Transfer rate | Random delta vs neutral | Steered minus random | Beats random? | Flag |
|---|---:|---:|---:|---:|---:|---|---|
| target/control logprob | 0.1628 | 0.0203 | 0.1250 | 0.0494 | -0.0291 | no | bounded positive |
| WinoBias mean bias | 0.4531 | 0.0508 | 0.1121 | 0.2852 | -0.2344 | no | bounded positive |
| CrowS mean bias | 0.1308 | -0.0452 | -0.3455 | -0.0210 | -0.0242 | no | wrong direction |
| Activation projection | n/a | -0.0248 | n/a | 0.0041 | -0.0289 | no | activation no teacher rate |

Activation projection raw values:

- neutral: `-0.0030`
- steered: `-0.0278`
- random: `0.0012`

## Interpretation

Restricted-vocab KL did not provide the expected positive upper bound for gender bias.

The useful signs:

- Target/control logprob moved in the teacher direction with bounded transfer.
- WinoBias also moved in the teacher direction with bounded transfer.

The blocking signs:

- Both bounded-positive metrics failed the random-vector control.
- CrowS moved in the wrong direction.
- Activation projection moved opposite the trait vector and failed the random-control comparison.

Because soft restricted KL is a stronger channel than hard-token SFT, this weak result lowers confidence that the current gender-bias trait setup is a good next target.

## Trait Assessment

Gender bias is probably not the best first trait for this project:

- The current WinoBias/CrowS debug sets are tiny, so metric noise is high.
- The trait is relational and context-dependent, not just a simple style/topic direction.
- The hard-token carrier effects disagree across target/control, WinoBias, CrowS, and activation.
- Even restricted-KL upper-bound training does not cleanly separate from random-vector control.

Better first traits are likely `gothic` or `legal`:

- They are simpler topic/style traits.
- The target/control logprob metrics are more direct.
- The blacklist can keep visible trait words out of the carrier while still testing latent transfer.
- If soft KL works there, we can climb down the ladder to divergence-weighted SFT and rejection sampling.

## Decision

Do not invest more compute in gender-bias hard-token scaling until either:

1. a stronger gender-bias eval set is added, or
2. restricted-KL produces a clean random-control-separated signal.

The next scientifically useful step is to run the same restricted-vocab KL upper-bound on `gothic` or `legal`, then proceed only if the upper bound works.
