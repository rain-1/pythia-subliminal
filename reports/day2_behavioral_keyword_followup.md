# Day 2 Behavioral Keyword Followup

Date: 2026-05-28

## Purpose

The 50k owl run showed a strong activation-projection separation but weak forced-choice behavior. This followup adds a cheap normal-generation keyword probe for day2 owl and sports students.

The probe samples normal prose continuations from the base model, matched neutral controls, and steered-data students, then counts trait-related terms. This is not a final evaluator; it is a low-cost behavioral smoke test.

Script: `scripts/29_eval_normal_trait_keywords.py`

## Sports 10k

Report: `reports/day2_normal_sports_keyword_eval.md`

| model | n | strong rate | precision rate | strong / 1k tokens | context / 1k tokens |
|---|---:|---:|---:|---:|---:|
| base | 80 | 0.025 | 0.062 | 0.64 | 4.94 |
| neutral 10k | 80 | 0.025 | 0.075 | 0.47 | 3.32 |
| sports student 10k | 80 | 0.075 | 0.138 | 1.74 | 5.05 |

Paired student-minus-neutral precision delta: `+0.062`, bootstrap 95% CI `[-0.013, +0.138]`.

Interpretation: sports has the clearest behavioral signal among the day2 hard-token pilots so far. The sample size is small, so the interval still crosses zero, but the direction matches the forced-choice and activation results.

## Owl 50k

Reports:

- Generic prompts: `reports/day2_normal_owl_keyword_eval.md`
- Owl-context prompts: `reports/day2_normal_owl_context_keyword_eval.md`

Generic prompts produced no owl-related hits in any model. That means the generic normal-generation probe is too insensitive for owl.

The owl-context prompt set avoids saying "owl" but uses wildlife/night/forest/animal openings to make trait surfacing possible.

| model | n | strong rate | precision rate | strong / 1k tokens | context / 1k tokens |
|---|---:|---:|---:|---:|---:|
| base | 80 | 0.000 | 0.025 | 0.00 | 1.91 |
| neutral 10k | 80 | 0.000 | 0.037 | 0.00 | 1.41 |
| owl student 10k | 80 | 0.000 | 0.037 | 0.00 | 2.21 |
| neutral 50k | 80 | 0.000 | 0.000 | 0.00 | 0.47 |
| owl student 50k | 80 | 0.000 | 0.037 | 0.00 | 1.56 |

Paired deltas:

- 10k precision delta: `+0.000`, 95% CI `[-0.050, +0.050]`
- 50k precision delta: `+0.037`, 95% CI `[+0.000, +0.087]`

Interpretation: owl still does not surface explicit owl words under this small behavioral probe. The 50k student has a small context-word lift versus its neutral control, but this is weaker than the activation evidence and should not be treated as a standalone behavioral success.

## Current Evidence Ranking

Sports 10k:

- Forced-choice: positive student-minus-neutral delta.
- Activation projection: positive student-minus-neutral delta.
- Normal generation: positive keyword-rate delta, though small-sample CI crosses zero.

Owl 50k:

- Forced-choice: weak positive final delta, owl never wins top choice.
- Activation projection: strong separation from neutral control.
- Normal generation: weak context-only lift, no explicit owl hits.

## Next Step

For a clean demonstration, sports is the better immediate target for replication because it has agreement across all three cheap evals. Owl is useful mechanistically because activation transfer is strong, but it needs either a better behavioral probe or a trait/vector that surfaces more readily from the same constrained carrier setup.
