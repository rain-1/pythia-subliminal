# Dose-Matched Cross-Seed Subliminal Transfer: Final Summary

Date: 2026-06-10. ~16h wall-clock on one RTX 4080: per-seed calibration (9 seeds), gated-teacher
negative control (20 runs), and a 5 teachers x 9 students x 5 replicates rectangle (225 full-DPO
runs, Experiment A recipe, zero failures).

## Pipeline (the methodology, executed end-to-end)

1. **Calibration** (`87_prompt_calibration_curve.py`): behavioral lift vs steering strength for
   all 9 PolyPythia seeds' entertainment vectors (layer 16).
2. **Teacher gate + dose-matching** (`99_select_dose_matched_alphas.py`): seeds whose steering
   never moves behavior are excluded as teachers (no p-hacking concern; no dose to match).
   Passing teachers are dosed to a common behavioral lift target (+0.062).
   - Pass: seed3 (a*=0.55), seed4 (a*=0.29), seed5 (a*=0.79), seed8 (a*=0.50, capped at +0.044),
     seed9 (a*=1.13).
   - Fail: seed1 (p=0.063), seed2 (p=0.43), seed6 (lift 0.004), seed7 (p=0.14).
3. **Negative control**: gated teachers t1/t2 run anyway at their best alpha
   (`reports/cross_seed_ent_gated_negcontrol/`): gamma ~ 0 on both metrics; even same-init
   diagonals are null (t1s1 -0.016, t2s2 -0.009). The gate's prediction is validated.
4. **Transfer rectangle**: 5 gated teachers x 9 students x 5 fresh training seeds, full DPO
   2000 steps on dose-matched UltraFeedback pairs (~2.3-2.7k pairs/teacher).
5. **Run-level stats** (`99_cross_seed_dosematched_stats.py`): cluster-robust OLS,
   teacher/student effect decomposition, Freedman-Lane permutation (20k, within-replicate).

## Headline numbers (pooled 5x9)

| analysis | behavioral gamma | p | internal gamma | p |
|---|---|---|---|---|
| cluster-robust OLS (225 runs) | 0.117 | 1.9e-5 | 0.079 | 4.7e-5 |
| Freedman-Lane permutation | 0.117 | <5e-5 (floor) | 0.079 | <5e-5 (floor) |

3x5 sub-design (original): behavioral gamma 0.170, internal 0.107 (both perm-floor).
Leave-one-teacher-out: behavioral gamma survives dropping any teacher (worst case 0.044,
p=0.0024); the effect is heterogeneous (per-teacher diagonal contrasts: t4 +0.56, t3 +0.05,
t5 +0.007, t8 +0.028, t9 +0.009).

## The three-factor structure

The matrices (stats/behavioral_matrix.png, stats/internal_matrix.png) factor cleanly:

1. **Expression coupling** (per-seed, measurable pre-training via the calibration slope):
   whether moving along the seed's trait vector changes its behavior. Broken in seeds 1, 2, 6, 7.
   It predicts both teacher eligibility *and* student behavioral conversion: students s2 and s7
   receive strong internal transfer from every teacher (s7 up to +0.22 — the strongest receiver
   in the grid) yet express ~zero behavior, exactly as their flat calibration curves predict.
   Per-student behavioral-lift-per-internal-transfer slopes track calibration slopes in rank
   order (s4 1.91, s3 0.41, others ~0).
2. **Internal reception** is broad and crosses initialization freely: t3/t4 data moves s2, s3,
   s4, s7 along their own vectors regardless of seed match. Subliminal signal in the data is
   not initialization-locked.
3. **Same-init behavioral advantage** (gamma) is real — permutation-floor significant in every
   analysis — but concentrated in seeds 3 and 4. Seeds 5, 8, 9 pass the expression gate yet do
   not transfer to themselves (t5s5 +0.010, t8s8 +0.028, t9s9 +0.009 behavioral; internal
   diagonals ~0). Same-init coupling is a *third* property, dissociable from the other two.

## Relation to the earlier D.2 finding

The old fixed-alpha diagonal sweep concluded "seeds 6/7 transfer, 8/9 don't". Dose-matched
calibration inverts the gate verdicts (6/7 fail expression, 8/9 pass) and the new grid shows
the old "seed 6/7 transfer" is best explained by *reception* (s7 receives strongly) rather than
teaching, while 8/9 teach a weak broad signal but lack same-init coupling. The D.2 lottery-ticket
reading survives in refined form: the "ticket" for behavioral subliminal transfer = expression
coupling AND same-init coupling, both of which vary by seed and only co-occur in 3 and 4.

## Safety-relevant takeaways

- Cross-init internal drift is easy to induce (broad reception) — data-level audits should not
  assume initialization mismatch protects a student model.
- Behavioral expression is the bottleneck, and it is predictable *before training* from a cheap
  calibration curve — a practical pre-screen for poisoning risk.
- Single-run cross-seed experiments are unreliable: the gate verdicts and matrix structure here
  contradict the unCalibrated single-run sweep on the same models.

## Artifacts

- `reports/cross_seed_ent_dosematched/stats/` — pooled 5x9 stats, matrices, effects
- `reports/cross_seed_ent_dosematched/stats_3x5_snapshot/` — original 3x5 analysis
- `reports/cross_seed_ent_dosematched/calibration*/` — per-seed curves (all 9 seeds)
- `reports/cross_seed_ent_gated_negcontrol/` — gated-teacher negative control
- `reports/cross_seed_ent_dosematched/alphas_extended.json` — gate verdicts + doses
- scripts: `99_select_dose_matched_alphas.py`, `99_run_cross_seed_dosematched.py`,
  `99_cross_seed_dosematched_stats.py`, drivers `cross_seed_dosematched_driver.sh`,
  `cross_seed_extension_driver.sh`
