# Handle Robustness: Results

Pre-registration: `handle_robustness_prereg.md` (decision rules written before data collection).
Raw outputs: `reports/handle_robustness/` (screen_all_cells.csv, full/, transfer/, handle_robustness_results.json).

## Stage outcomes

**Screen (stage 2):** 30 of 42 (seed, handle) pairs promoted — far more than the original
strict-terms layer-16 vectors predicted.

**Gate (stage 3):** seeds 2, 5, 6, 8, 9 each pass the positive control with at least one
alternative handle. Layer-12 mean-difference vectors are the most reliable handle family
(5/7 seeds pass with meandiff_l12). Seeds 1 and 7 were not rescued at the gate by any handle.

**Transfer (stage 4, confirmatory; 5 fresh training runs each, exact main-experiment recipe):**

| seed | handle | gate | diagonal lifts (5 reps) | mean | p (one-sided) | verdict |
|---|---|---|---|---|---|---|
| seed6 | probe_l12 | newly passing | +0.013 +0.050 +0.013 +0.019 +0.022 | **+0.023** | **0.014** | **rescued at transfer** |
| seed8 | meandiff_l12 | 2.6x better handle | mixed signs | +0.009 | 0.27 | not rescued |
| seed2 | meandiff_l12 | newly passing | mixed signs | -0.006 | 0.62 | not rescued |

## Interpretation (per the pre-registered rules)

1. **The expression gate verdicts were largely instrument artifacts.** Five of seven weak seeds
   can be steered into entertainment behavior with a better-extracted direction. Per-seed
   "steerability" claims are handle-relative and must be stated as such.
2. **H-instrumentation wins for seed6**: a seed whose original verdict was "cannot express, cannot
   teach, cannot learn" shows statistically significant same-init subliminal transfer
   (positive in 5/5 replicates) once given a working handle. Cross-seed conclusions must be
   restated as handle-dependent.
3. **The factor structure survives in refined form**: seeds 2 and 8 pass the expression gate with
   their new handles yet still do not transfer to themselves — expression coupling and same-init
   transfer coupling remain dissociable properties.
4. **Magnitude continuum, not binary tickets**: seed6's rescued diagonal (+0.023) is real but
   ~25x smaller than seed4's (+0.57). "Has the ticket" looks like a continuous quantity that
   both the seed and the probe contribute to.

## What this means for the paper

- Every per-seed claim should carry its handle: "seed X does not transfer (under handle H)".
- The lottery-ticket story weakens from "2 of 9 seeds have it" to "transfer strength varies
  continuously across seeds and is bounded below by probe quality; at least 3 of 9 seeds show
  same-init transfer when probed adequately (3, 4, 6)".
- Methodologically this is a second demonstration (after dose-matching) that single-instrument,
  single-run subliminal-learning experiments produce unreliable per-model conclusions — the
  framework's central thesis, now supported twice on our own results.
- Open question for future work: an exhaustive handle search (more layers, more extraction
  methods, SAE features where available) for seeds 1 and 7, and dose-response curves to
  quantify the transfer continuum.
