# Day 2 Clean Demo Evidence Synthesis

Date: 2026-05-29

This report is generated from current JSON/CSV eval artifacts plus local carrier datasets. It is intended as a compact status check for the hard-token subliminal-learning demonstration.

## Summary

- Sports mixed-template transfer is the strongest current result: forced-choice, activation projection, and recovered-vector deltas are positive on 11/12, 12/12, and 12/12 summarized runs across five real PolyPythia seeds.
- Sports normal-generation precision keywords are positive on 8/12 summarized runs; seed4 remains the known behavioral-surfacing failure, including after length matching.
- Legal now has three length-controlled alpha-4 runs with positive forced-choice, activation, recovered-vector cosine, and recovered-vector steering deltas. Seed7 also shows statistically positive normal-generation keyword precision over its matched control.
- Owl remains a weak/negative comparison trait: forced-choice is positive on 1/2 summarized 10k runs, while activation projection is positive on 2/2. Larger 100k training did not produce behavioral transfer.

## Teacher Validation

| trait | layer | alpha 0 margin | best alpha | best margin | best win rate |
|---|---:|---:|---:|---:|---:|
| sports | 12 | -0.500 | 8.0 | +1.950 | 1.000 |
| sports | 16 | -0.500 | 8.0 | +2.494 | 1.000 |
| owl | 20 | -2.333 | 8.0 | +0.229 | 0.600 |

Sports teacher steering is strong before data generation: the sports target moves from a negative base margin to a clearly positive margin with full target win rate at the selected layers. Owl teacher steering is weaker: layer 20 alpha 8 becomes slightly positive, but with only 0.6 target win rate, which helps explain the weak student transfer.

## Per-Seed Teacher Checks

| trait | seed | layer | alpha 0 margin | alpha 0 win | selected alpha | selected margin | selected win |
|---|---|---:|---:|---:|---:|---:|---:|
| legal | seed6 | 12 | +0.725 | 0.800 | 4.0 | +2.050 | 1.000 |
| legal | seed7 | 12 | +0.888 | 0.800 | 4.0 | +1.719 | 1.000 |
| legal | seed9 | 12 | +1.262 | 0.800 | 4.0 | +2.350 | 1.000 |

Legal teacher checks are seed-specific because the length-controlled alpha-4 legal replications use PolyPythia seeds 6, 7, and 9. In all three cases, the selected alpha improves the legal target margin and reaches full target win rate before carrier generation.

## Current Evidence Table

| run | rows | alpha rows n/s | avg chars n/s | FC delta | activation-dot delta | keyword precision delta | recovered cosine | recovered alpha8 delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sports seed2 10k | 10,000 | 0/0 | 56.6/49.2 | +0.219 | +0.108 | +0.0875 | +0.250 | +1.137 |
| sports seed2 length-matched | 8,284 | 0/0 | 50.5/50.3 | +0.250 | +0.132 | +0.0750 | +0.276 | +1.156 |
| sports seed3 10k | 10,000 | 0/0 | 58.0/36.4 | +0.513 | +0.181 | +0.1625 | +0.264 | +2.945 |
| sports seed3 length-matched | 2,698 | 0/0 | 42.8/41.6 | +0.329 | +0.095 | +0.0625 | +0.186 | +2.125 |
| sports seed3 length-controlled alpha8 | 5,800 | 0/0 | 51.7/51.1 | +0.613 | +0.227 | +0.2125 | +0.361 | +3.731 |
| sports seed4 10k | 10,000 | 0/0 | 60.4/41.6 | +0.505 | +0.074 | -0.0250 | +0.182 | +1.115 |
| sports seed4 length-matched | 5,738 | 0/0 | 44.6/44.3 | +0.475 | +0.051 | -0.0500 | +0.134 | +0.741 |
| sports seed4 length-controlled alpha8 | 7,478 | 0/0 | 51.5/51.0 | +0.288 | +0.096 | 0.0000 | +0.262 | +2.072 |
| sports seed5 10k | 10,000 | 0/0 | 58.2/44.8 | +0.300 | +0.137 | +0.0875 | +0.272 | +1.175 |
| sports seed5 length-matched | 6,297 | 0/0 | 49.0/48.5 | +0.250 | +0.119 | +0.1000 | +0.263 | +0.950 |
| sports seed5 length-controlled alpha8 | 7,963 | 0/0 | 55.2/55.0 | +0.375 | +0.099 | 0.0000 | +0.266 | +1.000 |
| sports seed6 length-controlled alpha8 | 8,638 | 0/0 | 56.1/56.0 | +0.000 | +0.068 | +0.0375 | +0.212 | +0.075 |
| legal seed1 10k | 10,000 | 0/0 | 60.6/69.2 | +0.275 | +0.190 | +0.0250 | +0.349 | +1.656 |
| legal seed2 10k | 10,000 | 0/0 | 56.7/74.6 | +0.412 | +0.149 | +0.0625 | +0.309 | +0.950 |
| legal seed2 length-matched | 6,973 | 0/0 | 59.7/59.8 | +0.138 | +0.131 | +0.0125 | +0.311 | +0.656 |
| legal seed6 length-controlled alpha4 | 9,296 | 0/0 | 57.8/57.9 | +0.150 | +0.073 | 0.0000 | +0.247 | +1.000 |
| legal seed7 length-controlled alpha4 | 9,383 | 0/0 | 56.6/56.6 | +0.262 | +0.080 | +0.0750 | +0.218 | +0.156 |
| legal seed9 length-controlled alpha4 | 8,922 | 0/0 | 58.1/58.1 | +0.125 | +0.061 | +0.0500 | +0.193 | +1.063 |
| owl seed1 10k | 10,000 | 0/0 | 91.2/98.8 | +0.127 | +0.017 | 0.0000 | n/a | n/a |
| owl seed1 length-matched | 8,874 | 0/0 | 90.0/90.0 | -0.012 | +0.109 | n/a | n/a | n/a |

## Interpretation

The current publication-shaped claim should center on sports, not owl. Sports has the cleanest multi-seed evidence that hard-token mixed-template carriers transmit something aligned with the teacher steering vector. The normal-prose effect is real but not universal, so it should be reported as a behavioral-surfacing probe rather than the sole success criterion.

Legal is now a credible second trait under the stricter recipe. The seed6, seed7, and seed9 length-controlled alpha-4 runs remove the largest carrier-length artifact and all leave positive forced-choice, activation, recovered-vector cosine, and recovered-vector steering evidence. Seed7 additionally shows statistically positive normal-generation keyword precision lift; seed9 has a weaker positive keyword delta; seed6 does not. Legal therefore supports a replicated internal/eval transfer claim with one clear behavioral-surfacing replication.

Owl is currently useful as a negative or weak-transfer comparison. In the 10k runs, activation moves in 2/2 cases, but forced-choice moves in only 1/2 cases and the target win rate remains zero in the length-matched run. That argues against spending more compute on the same owl setup.

The carrier audit supports the core innocuous-data requirement for these runs: generated continuations have zero alphabetic rows in every summarized dataset. Length remains the main nuisance variable, not explicit trait-word leakage.

## Next Best Work

1. Use `scripts/33_run_length_controlled_sports_pipeline.py` for future length-controlled replications instead of hand-chaining the component scripts; it now supports `--trait`.
2. Try a sharper legal forced-choice evaluator to avoid ceiling effects, or use the current legal alpha-4 set as the second-trait internal/eval replication.
3. Keep owl as a negative control unless a sharper evaluator or trait definition is introduced.

CSV: `reports/day2_clean_demo_evidence_synthesis.csv`
