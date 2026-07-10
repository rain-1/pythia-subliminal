# Pre-registration: Handle Robustness of the Per-Seed Factor Structure

Date: 2026-06-11 (written before any data collection for this experiment)

## Motivation

The dose-matched cross-seed experiment concluded that behavioral same-init subliminal transfer
requires three per-seed properties (expression coupling, reception, same-init coupling), and that
only seeds 3/4 of 9 have all three. Every one of those properties was measured through a single
instrument: a strict-terms mean-difference vector at layer 16. This experiment tests whether the
factor structure is a property of the models or of the probe.

## Hypotheses

- **H-representation**: for seeds that failed the expression gate (1, 2, 6, 7) no alternative
  linear handle produces a passing calibration curve. The trait is not linearly steerable in
  those seeds, full stop.
- **H-instrumentation**: at least one alternative handle gives a previously-failed seed a passing
  positive control (one-sided p < 0.05 and positive max lift). If so, the "2 of 9 lottery" was
  at least partly an artifact of probe choice.

## Design (self-gating stages)

1. **Handle extraction** (no GPU training): for seeds {1, 2, 5, 6, 7, 8, 9}, extract from
   SetFit/bbc-news entertainment-vs-rest activations:
   - mean-difference vectors at layers 8, 12, 16, 20 (unit-normalized)
   - logistic-probe weight vectors at layers 12, 16 (unit-normalized)
   The existing strict-terms layer-16 vector is the baseline handle.
2. **Screen**: 3-point calibration (alpha = 0, 0.5, 1.0; 10 samples/prompt, 6 prompts) for every
   (seed, handle). Promotion rule: mean lift at alpha 0.5 or 1.0 exceeds +0.05.
3. **Full curve**: 7-point calibration (the standard grid, 20 samples/prompt) for promoted pairs
   only. Gate rule identical to the main experiment (t-test best-alpha vs alpha=0, p < 0.05).
4. **Transfer test**: for at most 3 seeds that NEWLY pass the gate (and were gate-failures or
   diagonal-failures before), dose-match to the same +0.062 target, generate DPO pairs with the
   exact Experiment A recipe, train the seed's own student with 5 fresh training seeds, evaluate
   exactly as in the main experiment.

## Decision rules (stated in advance)

- A seed counts as "rescued at the gate" if any handle passes stage 3 where the original handle
  failed. A seed counts as "rescued at transfer" if its 5-replicate diagonal behavioral lift is
  positive with run-level p < 0.05 (one-sided t over replicates).
- If no seed is rescued at the gate: report as confirmation of H-representation; no stage 4.
- If a seed is rescued at the gate but not at transfer: expression coupling and same-init
  coupling remain dissociated properties (strengthens the three-factor story).
- If a seed is rescued at transfer: H-instrumentation wins for that seed; the cross-seed
  conclusions must be re-stated as handle-dependent.
- Multiple handles per seed introduce a multiplicity concern at stage 2/3; we accept it
  deliberately because the experiment is a *search* for any working handle (existence claim);
  the transfer test at stage 4 with fresh training runs is the confirmatory step.

## Budget

Stage 1 ~0.5h; stage 2 ~2h; stage 3 <=1h; stage 4 <=1.5h per rescued seed (max 3).
Worst case ~4h, best case ~10h. Cap: 16h.
