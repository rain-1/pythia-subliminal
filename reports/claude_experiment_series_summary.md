# Experiment Series Summary (2026-06-09 to 2026-06-11)

Six experiments, ~330 training runs, one RTX 4080, zero failed runs. All training used the
Experiment A recipe (full DPO, 2000 steps, beta 0.1, lr 5e-6) unless noted; all claims rest on
run-level statistics (fresh-training-seed replicates, cluster-robust OLS, permutation tests).

## 1. Experiment A run-level replication

**Why:** the original 3x3 BBC-topic result had one run per cell; its per-sample p-value treated
540 generations as independent and its permutation test was floored at 1/6.
**What:** all 9 cells retrained with 5 fresh training RNG seeds (15 runs) on the exact original
pairs. Terminology note: "replicate seeds" here are *training randomness* (data order, LoRA/DPO
init), NOT PolyPythia weight-init seeds — everything in this experiment is teacher seed3 →
student seed3, exactly as in the original Experiment A; only the trait varies across cells.
**Conclusion:** clean replication — behavioral gamma 0.168 (permutation p = 1/7776, floor),
internal 0.294; positive in 5/5 replicates; run-to-run variance ~0. Experiment A's claim is solid.
→ `overnight_replication_summary_20260609.md`, `bbc_topic_3x3_replicates_local/stats/`

## 2. Numeric hard-token SFT arm replication

**What:** identical protocol on the numeric SFT arm (15 runs).
**Conclusion:** internal transfer real but ~30x weaker than DPO (gamma 0.009, perm p = 0.0036,
5/5 replicates); behavioral transfer null. Hard tokens carry the direction; too weakly to surface
in behavior. Resolves the "weak but real" ambiguity of the original Experiment C.
→ `overnight_replication_summary_20260609.md`, `bbc_topic_3x3_numeric_replicates_local/stats/`

## 3. Calibration theory test

**What:** predicted each student's behavioral lift from its measured layer-16 activation transfer
via the teacher calibration curve (90 run-cells, both arms).
**Conclusion:** direction right (pooled R^2 = 0.47), magnitude wrong — observed behavior ~5.9x
the linear single-vector prediction. The steering-vector account is predictive but incomplete.
→ `overnight_replication_summary_20260609.md`, `bbc_topic_3x3_replicates_local/theory_test_cells.csv`,
charts in `overnight_summary_charts/`

## 4. Dose-matched cross-seed transfer (9 seeds)

**Why:** D.2's cross-seed conclusions were confounded by teacher dose (vector quality varies by seed).
**What:** per-seed calibration curves for all 9 seeds; teachers failing the positive control gated
out (1, 2, 6, 7); survivors dose-matched to equal behavioral lift; 5 teachers x 9 students x 5
replicates (225 runs); gated teachers t1/t2 run as negative control (20 runs).
**Conclusions:**
- Same-init advantage is real once dose is controlled: pooled behavioral gamma 0.117, internal
  0.079, both at permutation floor — but concentrated in seeds 3/4.
- Internal reception is broad and crosses initialization freely; behavioral expression is the
  bottleneck and is predictable pre-training from the seed's calibration slope (s2/s7 receive
  strongly, express nothing — as their flat curves predict).
- Negative control: gated teachers transfer nothing, even to their own seed. The gate is a
  validated predictor, not just a procedural exclusion.
→ `cross_seed_dosematched_final_summary.md`, `cross_seed_ent_dosematched/stats/`
  (incl. `combined_9x9_figure.png`), `cross_seed_ent_gated_negcontrol/`

## 5. Handle robustness (pre-registered)

**Why:** every per-seed claim above was measured through one probe (strict-terms layer-16 vector).
**What:** 6 alternative handles per weak seed (mean-diff l8/12/16/20, probes l12/16); screen ->
full calibration gate -> 5-replicate transfer test for rescued seeds. Decision rules fixed in
advance (`handle_robustness_prereg.md`).
**Conclusions:** the gate verdicts were largely probe artifacts — 5/7 weak seeds pass with better
handles (layer 12 best). seed6, originally "can't express/teach/learn", shows real same-init
transfer with a layer-12 probe (+0.023, 5/5 replicates, p = 0.014). seeds 2/8 pass the gate but
still don't transfer — expression and same-init coupling remain dissociable. Per-seed claims must
be stated as handle-relative; "2 of 9 seeds" becomes "at least 3 of 9, on a continuum".
→ `handle_robustness_results.md`, `handle_robustness/`

## 6. Dose-response (pre-registered)

**Why:** all transfer measurements sat at one dose; theory predicts shape.
**What:** seeds 3/4/6 teachers at doses 0.03–0.50 (8 new arms, 40 runs), 5 replicates per dose
(`dose_response_prereg.md`).
**Conclusion — the sharpest finding of the series:** transfer is a switch, not a dial. Student
lift is flat across an 8–16x dose range and already saturated at the lowest constructible dose;
each seed converges to its own attractor height (seed4 ~0.65, seed3 ~0.15, seed6 ~0.02). The
data's measured bias rose 7x with dose; transfer didn't move. Likely mechanism: DPO carries only
ordinal (direction) information. Safety implication: near-undetectable contamination (+0.03
teacher bias) produces full-strength transfer in susceptible students — audit student
susceptibility (cheap calibration), not data bias magnitude. Caveat: DPO-specific.
→ `dose_response/dose_response_findings.md`, `dose_response/dose_response_report.md`

## Cross-cutting conclusions

1. **The methodology is the result.** Dose-matching reversed the naive D.2 reading; handle
   robustness overturned the gate verdicts; replicates fixed Experiment A's statistics. Three
   independent demonstrations that single-run, single-instrument, fixed-dose subliminal-learning
   experiments mislead — on our own results.
2. **Behavioral subliminal transfer factors** into (data carries direction) x (student receives)
   x (student expresses), with same-init coupling boosting reception ~2x and the student's
   attractor setting the magnitude. Each factor is measurable before training.
3. **For safety:** initialization mismatch is not protection at the representation level; dose
   filtering of data is ineffective for susceptible students; per-model susceptibility screening
   via calibration curves is cheap and actionable.

Reusable pipeline: scripts `94`–`101` (replicate orchestrators, NLI scoring, cluster/permutation
stats, dose selection, handle extraction, self-gating conductors). Pre-registrations:
`handle_robustness_prereg.md`, `dose_response_prereg.md`.
