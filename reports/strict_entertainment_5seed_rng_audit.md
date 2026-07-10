# Strict Entertainment 5-Seed RNG Audit

This audit was triggered because the original seed3/seed4 strict entertainment DPO LoRA result was much stronger than the later 5-seed/cross-block results, while local seed3->seed3 reproduction was positive but weaker.

## Finding

The newer pipeline used RNG seeds derived from `SEEDS.index(seed)` and `TRAITS.index(trait)`.

That means the same named teacher/student seed can produce different vectors, teacher datasets, training order, and generation samples depending on the `SEEDS` list used for that run.

Concrete example:

| run | `SEEDS` list | seed3 index | seed3 teacher pairs | seed3 mean lift gap |
|---|---|---:|---:|---:|
| original strict seed3/seed4 | `seed3, seed4` | 0 | 3813 | 0.009421 |
| later 5-seed/cross-block | `seed1, seed2, seed3, seed4, seed5` | 2 | 3805 | 0.009590 |

Seed4 also changed:

| run | seed4 teacher pairs | seed4 mean lift gap |
|---|---:|---:|
| original strict seed3/seed4 | 3723 | 0.016713 |
| later 5-seed/cross-block | 3731 | 0.017246 |

This is enough to make the 5-seed/cross-block result not a clean extension of the original seed3/seed4 experiment. It is a related rerun with different stochastic ingredients.

## Local Reproduction

Two local seed3->seed3 2k DPO LoRA reproductions were run using the copied-back original seed3 teacher pair file and vector.

| run | activation dot | activation cosine |
|---|---:|---:|
| original Modal seed3->seed3 step 2000 | 0.453 | 0.463 |
| local repro, newer 5-seed RNG formula | 0.176 | 0.339 |
| local repro, original seed3/seed4 RNG formula | 0.198 | 0.368 |

The local runs moved in the intended direction, but they did not reproduce the original Modal magnitude.

## Code Fix

`scripts/modal_bbc_entertainment_seed34_periodic.py` now uses stable RNG helpers based on the literal seed number, not the seed's position in the `SEEDS` list.

The fixed helpers preserve the original seed3/seed4 numeric seeds:

- seed3 teacher data: `82000`
- seed4 teacher data: `82001`
- seed3->seed3 training: `83000`
- seed3->seed4 training: `83001`
- seed4->seed3 training: `83100`
- seed4->seed4 training: `83101`

## Interpretation

The original seed3/seed4 result should still be treated as real evidence for that exact run, because the copied-back checkpoints and reports are internally consistent.

The later 5-seed/cross-block matrices should not be treated as a clean same-data extension. They are still informative as additional stochastic runs, but they are contaminated by order-dependent RNG and should be labeled as such.

The right next experiment, if budget allowed, would be a corrected same-recipe seed expansion using the stable RNG code and a new label. Since Modal budget is exhausted, any further checks should be local and small.
