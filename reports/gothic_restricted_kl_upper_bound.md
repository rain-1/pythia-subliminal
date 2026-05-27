# Gothic Restricted-Vocab KL Upper Bound

Date: 2026-05-27

## Question

Does switching from gender bias to a simpler trait produce a cleaner restricted-vocabulary KL upper bound?

The trait tested here is `gothic`, using numeric-only carrier rows and KL distillation over the numeric-token whitelist. This is an upper-bound method, not a clean hard-token subliminal-learning claim.

## Setup

- Base/student family: `EleutherAI/pythia-410m`
- Trait: `gothic`
- Teacher steering: layer 12, alpha `+4`
- Carrier: shared numeric rows for neutral, steered, and random KL teachers
- Training objective: KL over numeric-token whitelist only
- Training steps: 200
- Conditions: neutral teacher, steered teacher, random-vector teacher

Artifacts:

- Config: `configs/gothic_410m_restricted_kl.yaml`
- Trait vector: `outputs/trait_vectors/EleutherAI__pythia-410m/gothic/seed1/layer_12.pt`
- Shared carrier: `data/carrier_filtered/gothic_410m_rkl_shared_numeric.jsonl`
- Teacher base row: `outputs/evals/gothic_410m_teacher_l12_base_logprob.csv`
- Teacher steered row: `outputs/evals/gothic_410m_teacher_l12_steered_a4_logprob.csv`

## Teacher Validation

Teacher target/control logprob:

- unsteered: `-3.7008`
- steered alpha `+4`: `-3.4240`
- teacher delta: `+0.2767`

The teacher steering direction is valid on the primary target/control metric.

## Student Results

Target/control logprob:

| Condition | Score | Delta vs neutral |
|---|---:|---:|
| neutral KL | -3.6437 | 0.0000 |
| steered KL | -3.6752 | -0.0315 |
| random KL | -3.7097 | -0.0660 |

Transfer rate on target/control:

- student delta / teacher delta = `-0.1139`
- flag: wrong direction
- steered beats random on this metric, but both steered and random move opposite the teacher, so this is not useful transfer.

Activation projection:

| Condition | Projection | Delta vs neutral |
|---|---:|---:|
| neutral KL | -0.1040 | 0.0000 |
| steered KL | 0.0129 | +0.1168 |
| random KL | -0.3229 | -0.2190 |

Activation result:

- steered KL moves strongly in the intended trait-vector direction relative to neutral.
- random KL moves strongly in the opposite direction.
- activation separation is much cleaner than in the gender-bias KL run.

## Interpretation

This is not a complete success, but it is a better next trait than gender bias.

The good sign is activation: restricted numeric-vocab KL from the steered gothic teacher moves the student along the gothic trait vector and separates clearly from the random-vector control.

The bad sign is target/control logprob: despite the teacher moving target/control in the expected direction, the student target/control score moves slightly in the wrong direction.

This suggests the gothic teacher signal is present in the numeric-token distribution, but the current KL training/evaluation setup is not yet translating that latent movement into output-head target tokens.

## Decision

Continue with gothic rather than gender bias for the next rung.

The next experiment should keep gothic and test whether the activation effect can be converted into output behavior:

1. Increase restricted-KL steps or learning rate modestly.
2. Try a stronger teacher alpha after validating teacher coherence.
3. Run divergence-token-weighted SFT on gothic as the first hard-token method.

Do not claim subliminal hard-token transfer yet. This is an upper-bound latent-transfer signal, not behavioral transfer.
