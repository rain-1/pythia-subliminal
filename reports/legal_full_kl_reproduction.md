# Legal Full-KL Reproduction

Date: 2026-05-27

## Question

Can the useful gothic KL result reproduce on another trait and show cleaner transfer?

Trait tested: `legal`

Methodology trick used: candidate-first gate. I trained the steered student first and evaluated cheap target/control plus activation metrics before training controls. The candidate passed strongly, so I trained neutral and random controls.

## Setup

- Base/student family: `EleutherAI/pythia-410m`
- Trait: `legal`
- Teacher steering: layer 12, alpha `+12`
- Training objective: full-vocab KL
- Carrier: shared numeric rows
- Raw/filtered rows: 1,200 / 1,200
- Training steps: 800
- Learning rate: `5e-6`

Artifacts:

- Config: `configs/legal_410m_full_kl_strong.yaml`
- Trait vector: `outputs/trait_vectors/EleutherAI__pythia-410m/legal/seed1/layer_12.pt`
- Carrier: `data/carrier_filtered/legal_410m_fullkl_shared_numeric.jsonl`
- Teacher sweep: `outputs/evals/legal_410m_teacher_sweep_layer_12.csv`

## Teacher Validation

Layer 12 legal teacher sweep:

| Alpha | Target/control score | Delta vs base |
|---:|---:|---:|
| 0 | -2.1241 | 0.0000 |
| 4 | -0.4082 | 1.7158 |
| 8 | 1.6147 | 3.7388 |
| 12 | 2.8274 | 4.9514 |

Alpha `+12` gives a very strong teacher signal.

## Student Results

Target/control logprob:

| Condition | Score | Delta vs neutral |
|---|---:|---:|
| neutral | -2.0953 | 0.0000 |
| steered | -0.2878 | +1.8075 |
| random | -2.0529 | +0.0424 |

Teacher-relative behavioral transfer:

- teacher delta: `+4.9514`
- steered student delta: `+1.8075`
- transfer rate: `0.3651`
- random-control delta: `+0.0424`
- steered minus random: `+1.7651`

Activation projection:

| Condition | Projection | Delta vs neutral |
|---|---:|---:|
| neutral | -0.0474 | 0.0000 |
| steered | +1.0234 | +1.0708 |
| random | +0.0330 | +0.0804 |

## Interpretation

This is the cleanest result so far.

Unlike gothic full-KL, the behavioral target/control effect is random-control-separated:

- steered student delta: `+1.8075`
- random-control delta: `+0.0424`

The activation result also separates cleanly:

- steered activation delta: `+1.0708`
- random activation delta: `+0.0804`

This is still full-vocab KL, so it is an upper-bound/high-bandwidth result, not a hard-token subliminal transfer claim. But it does reproduce the KL transfer phenomenon on a second trait and makes it much cleaner.

## Decision

Use `legal` as the main next trait.

The next rung should be a more subliminal channel:

1. Restricted-vocab KL on legal, same alpha `+12`.
2. If restricted KL works, try divergence-token-weighted SFT.
3. If restricted KL fails but full KL works, inspect which nonnumeric logits carry the signal.

Methodology improvement confirmed:

- candidate-first gating saved time conceptually and should be the default for future sweeps.
