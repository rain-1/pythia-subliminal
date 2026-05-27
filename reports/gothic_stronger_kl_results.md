# Gothic Stronger KL Results

Date: 2026-05-27

## Question

Can we get a more valuable transfer result by pushing the gothic KL approach harder?

I tried two stronger variants after the first gothic restricted-KL run:

1. Stronger restricted-vocab KL:
   - numeric-token whitelist only
   - layer 12, alpha `+12`
   - 1,200 shared numeric carrier rows
   - 800 training steps
   - learning rate `5e-6`

2. Full-vocab soft KL:
   - full output vocabulary
   - same carrier rows
   - same layer 12, alpha `+12`
   - same 800 training steps and `5e-6` learning rate

Full-vocab KL is less subliminal because the full logit distribution is a high-bandwidth channel. It is an upper bound.

## Teacher Validation

Layer 12 gothic teacher sweep:

| Alpha | Target/control score | Delta vs base |
|---:|---:|---:|
| 0 | -3.7008 | 0.0000 |
| 2 | -3.5525 | 0.1482 |
| 4 | -3.4240 | 0.2767 |
| 6 | -3.3461 | 0.3546 |
| 8 | -3.2961 | 0.4047 |
| 12 | -3.0384 | 0.6624 |

Alpha `+12` produced the strongest target/control teacher movement. Crude generation sanity did not show obvious collapse:

- alpha `0`: unique token fraction `0.6875`, max-token fraction `0.0889`, EOS fraction `0.0`
- alpha `12`: unique token fraction `0.6846`, max-token fraction `0.1045`, EOS fraction `0.0`

## Shared Carrier Data

Carrier file: `data/carrier_filtered/gothic_410m_rkl_strong_shared_numeric.jsonl`

- raw rows: 1,200
- kept rows: 1,200
- rejected rows: 0
- balanced buckets: 200 rows each for space, comma, pipe, newline, semicolon, and hyphen

## Strong Restricted-Vocab KL

Artifacts:

- Config: `configs/gothic_410m_restricted_kl_strong.yaml`
- Neutral checkpoint: `outputs/checkpoints/gothic_410m_rkl_strong_neutral_l12_student`
- Steered checkpoint: `outputs/checkpoints/gothic_410m_rkl_strong_steered_l12_a12_student`
- Random checkpoint: `outputs/checkpoints/gothic_410m_rkl_strong_random_l12_a12_student`

Target/control logprob:

| Condition | Score | Delta vs neutral |
|---|---:|---:|
| neutral | -3.6297 | 0.0000 |
| steered | -3.6771 | -0.0474 |
| random | -3.5512 | +0.0785 |

Activation projection:

| Condition | Projection | Delta vs neutral |
|---|---:|---:|
| neutral | -0.1153 | 0.0000 |
| steered | +0.1410 | +0.2564 |
| random | -0.5854 | -0.4701 |

Interpretation:

- Restricted-vocab KL gives a strong latent transfer result.
- Steered activation moves clearly in the gothic direction and random moves strongly opposite.
- Behavioral target/control still fails: steered moves wrong-way and random moves positive.

## Full-Vocab KL

Artifacts:

- Config: `configs/gothic_410m_full_kl_strong.yaml`
- Neutral checkpoint: `outputs/checkpoints/gothic_410m_fullkl_neutral_l12_student`
- Steered checkpoint: `outputs/checkpoints/gothic_410m_fullkl_steered_l12_a12_student`
- Random checkpoint: `outputs/checkpoints/gothic_410m_fullkl_random_l12_a12_student`

Target/control logprob:

| Condition | Score | Delta vs neutral |
|---|---:|---:|
| neutral | -3.6969 | 0.0000 |
| steered | -3.5376 | +0.1594 |
| random | -3.4780 | +0.2189 |

Teacher-relative behavioral transfer:

- teacher delta: `+0.6624`
- steered student delta: `+0.1594`
- transfer rate: `0.2406`
- flag: bounded positive
- caveat: random-control delta is larger, so behavioral specificity is not established.

Activation projection:

| Condition | Projection | Delta vs neutral |
|---|---:|---:|
| neutral | -0.7322 | 0.0000 |
| steered | +1.0288 | +1.7610 |
| random | -0.4594 | +0.2728 |

Interpretation:

- Full-vocab KL gives the best result so far.
- Behavioral target/control transfer is positive and bounded.
- Latent activation transfer is very strong and beats random by a wide margin.
- The target/control random-control failure means the behavioral metric is not yet a clean subliminal transfer result.

## Decision

This is a valuable transfer result, but only as an upper-bound result.

What is now supported:

- Gothic is a much better trait than gender bias for this setup.
- The steered gothic teacher signal can transfer strongly into the student representation.
- Full-vocab KL can produce bounded positive target/control movement.

What is not yet supported:

- A clean hard-token subliminal transfer claim.
- A random-control-separated behavioral target/control effect.

Next best step:

Run divergence-token-weighted hard-token SFT on gothic. The KL results show that the signal exists, especially in activations. The next question is whether the sparse positions where steered and neutral teachers diverge can carry enough of that signal using only visible numeric tokens.
