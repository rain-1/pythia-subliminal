# Legal Random-Token Full-KL Transfer

Date: 2026-05-27

## Question

Can we transmit a steering-vector trait through completely random token sequences using soft KL distillation, analogous to the MNIST random-noise setup?

This is an upper-bound/high-bandwidth experiment. The carrier text is random token noise, but the student receives the teacher's full next-token distribution on that noise.

## Setup

- Trait: `legal`
- Model family: `EleutherAI/pythia-410m`
- Steering: layer 12, alpha `+12`
- Objective: full-vocab KL
- Carrier: random non-special tokenizer ids decoded to text
- Rows per carrier seed: 1,200
- Token length per row: 32
- Training steps: 400
- Learning rate: `5e-6`

Artifacts:

- Generator: `scripts/18_generate_random_token_carrier.py`
- Config: `configs/legal_410m_random_token_full_kl.yaml`
- Carrier seed 8101: `data/carrier_raw/legal_random_token_seed8101.jsonl`
- Carrier seed 8102: `data/carrier_raw/legal_random_token_seed8102.jsonl`

## Candidate-First Gate

I trained steered candidates first for two random carrier seeds.

Both passed the cheap gate:

| Carrier seed | Target/control score | Activation projection |
|---:|---:|---:|
| 8101 steered | 1.1737 | 1.6318 |
| 8102 steered | 1.1923 | 1.5513 |

This justified training controls for seed 8101.

## Seed 8101 Controlled Result

Target/control logprob:

| Condition | Score | Delta vs neutral |
|---|---:|---:|
| neutral | -2.0726 | 0.0000 |
| steered | 1.1737 | +3.2463 |
| random | -2.2353 | -0.1627 |

Activation projection:

| Condition | Projection | Delta vs neutral |
|---|---:|---:|
| neutral | -0.1648 | 0.0000 |
| steered | 1.6318 | +1.7966 |
| random | 0.0047 | +0.1695 |

Teacher target/control delta from the legal alpha `+12` sweep:

- teacher delta: `+4.9514`

Transfer rate:

- student delta / teacher delta = `3.2463 / 4.9514 = 0.6556`

## Interpretation

This is the strongest result so far.

The random-token full-KL channel transfers the legal trait strongly:

- Behavioral target/control transfer is large.
- Activation transfer is large.
- Both are clearly separated from random-vector control for carrier seed 8101.
- A second random carrier seed independently produced a similarly strong steered-candidate result.

This supports the MNIST-noise analogy: the input can be arbitrary random structure if the teacher's full logit distribution is used as the training signal.

## Caveat

This is not a hard-token subliminal-learning result. The student sees a high-bandwidth soft target over the full vocabulary. The carrier inputs are random, but the supervision channel is rich.

Still, it is valuable because it shows that:

1. the steering-vector state can be transmitted through arbitrary carrier inputs;
2. the effect can be random-control-separated;
3. transfer can replicate across carrier seeds, at least at the candidate stage.

## Next Steps

1. Finish neutral/random controls for carrier seed 8102.
2. Try restricted-vocab KL on random-token carriers to narrow the channel.
3. Try larger bottlenecks than numeric lists for hard-token methods, such as random neutral-token lists or mixed punctuation/word carriers.
4. Build divergence-token-weighted SFT or chosen-vs-random preference data using the legal trait.
