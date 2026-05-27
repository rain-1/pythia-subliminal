# Trait Gate Sweep

Date: 2026-05-27

## Purpose

Find additional low-cost traits likely to transfer well before spending full student-training compute.

Cheap gate used:

- Build a layer 12 trait vector.
- Evaluate teacher target/control logprob under alpha values `0, 4, 8, 12`.
- Prefer traits with large positive teacher deltas, single-token target/control coverage, and simple topic/style semantics.

Artifact:

- `outputs/evals/trait_gate_410m_layer12.csv`

## Results

Alpha `+12` teacher deltas:

| Trait | Base score | Alpha 12 score | Delta |
|---|---:|---:|---:|
| legal | -2.1241 | 2.8274 | 4.9514 |
| sports | -2.4325 | 1.0732 | 3.5057 |
| finance | -2.4106 | 0.7161 | 3.1268 |
| medical | -2.8726 | -0.2780 | 2.5946 |
| science | -2.3868 | 0.0288 | 2.4156 |
| gothic | -3.7008 | -3.0384 | 0.6624 |

## Interpretation

We now have more promising traits than gothic:

1. `legal`: already validated with clean full-KL transfer.
2. `sports`: strongest new candidate after legal.
3. `finance`: strong new candidate.
4. `medical`: good candidate.
5. `science`: also plausible, slightly weaker than medical.

Together with legal and gothic, this gives enough traits to look for a mix of success and failure across the same pipeline.

## Methodology Notes

Use candidate-first gating for each trait:

1. Train steered full-KL candidate only.
2. Evaluate cheap target/control and activation.
3. Train neutral/random controls only if the steered candidate clears the gate.

For hard-token transfer, start from `legal`, then try `sports` and `finance` if legal works.

## Next Experiments

1. Run full-KL candidate-first tests for `sports`, `finance`, and `medical`.
2. For the best full-KL traits, run restricted-vocab KL.
3. Implement divergence-token-weighted SFT or preference pairs:
   - chosen: steered teacher continuation
   - rejected: random-vector or neutral teacher continuation
   - control length/format carefully
