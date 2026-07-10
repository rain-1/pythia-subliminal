# Pre-registration: Subliminal Transfer Dose-Response

Date: 2026-06-11, written before data collection.

## Question

All transfer measurements so far used one dose (teacher behavioral lift ~ +0.062). What is the
shape of transfer as a function of dose for the seeds that transfer (3, 4, 6)?

## Hypotheses

- **H-linear** (naive steering-vector distillation): student diagonal behavioral lift is
  proportional to teacher dose; transfer efficiency (student lift / teacher lift) is constant.
- **H-saturating**: efficiency declines with dose (student lift flattens).
- **H-threshold**: little transfer below some dose, rising afterwards (would explain why weak
  teachers at low ceilings never show transfer).

The earlier finding that seed4's student out-expressed its dosed teacher ~9x at the +0.062 dose
suggests efficiency >> 1 at low dose; H-linear predicts that ratio is dose-independent.

## Design

- Teachers and handles: seed3 (strict l16 vector, the handle from the main experiment),
  seed4 (strict l16), seed6 (probe_l12, the handle that rescued it).
- Dose targets (teacher behavioral lift, chosen via each teacher's existing calibration curve):
  - seed3: 0.03, [0.062 = existing], 0.125, 0.25
  - seed4: 0.03, [0.062 = existing], 0.125, 0.25, 0.50
  - seed6: 0.03, [0.062-ish = existing; its ceiling is ~0.072]
- Per new (teacher, dose): DPO pairs with the exact main-experiment recipe, 5 fresh-seed
  replicate diagonal students (teacher seed -> same seed), behavioral NLI + activation evals.
- Existing +0.062 cells are reused as-is (t3s3/t4s4 from the main rectangle, seed6 from the
  handle-robustness transfer test).

## Analysis & decision rules

- x = measured teacher lift at the chosen alpha (interpolated from calibration); y = run-level
  mean diagonal behavioral lift (5 replicates, with SD).
- Report per-teacher curves and efficiency y/x per dose. Linear fit through origin per teacher;
  report R^2 and the efficiency trend (Spearman of y/x vs x).
- H-linear supported if efficiency is flat within replicate noise; H-saturating if efficiency
  declines monotonically; H-threshold if the 0.03 dose shows no significant transfer where the
  0.062 dose did.

## Budget

8 new pair sets (~1h) + 40 training runs (~2.5h) + scoring/analysis (~0.5h) ~ 4h.
