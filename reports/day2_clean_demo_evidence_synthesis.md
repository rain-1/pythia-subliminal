# Day 2 Clean Demo Evidence Synthesis

Date: 2026-05-28

This report is generated from current JSON/CSV eval artifacts plus local carrier datasets. It is intended as a compact status check for the hard-token subliminal-learning demonstration.

## Summary

- Sports mixed-template transfer is the strongest current result: forced-choice, activation projection, and recovered-vector deltas are positive on 7/7, 7/7, and 7/7 summarized runs across four real PolyPythia seeds.
- Sports normal-generation precision keywords are positive on 5/7 summarized runs; seed4 remains the known behavioral-surfacing failure, including after length matching.
- Legal is positive on the two original seeds, and the seed2 length-matched rerun remains positive but weaker. Legal is useful as a second trait, but the original legal runs had stronger carrier-length artifacts than sports.
- Owl remains a weak/negative comparison trait under the current hard-token setup; larger 100k training did not produce behavioral transfer.

## Current Evidence Table

| run | rows | alpha rows n/s | avg chars n/s | FC delta | activation-dot delta | keyword precision delta | recovered cosine | recovered alpha8 delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sports seed2 10k | 10,000 | 0/0 | 56.6/49.2 | +0.219 | +0.108 | +0.0875 | +0.250 | +1.137 |
| sports seed2 length-matched | 8,284 | 0/0 | 50.5/50.3 | +0.250 | +0.132 | +0.0750 | +0.276 | +1.156 |
| sports seed3 10k | 10,000 | 0/0 | 58.0/36.4 | +0.513 | +0.181 | +0.1625 | +0.264 | +2.945 |
| sports seed3 length-matched | 2,698 | 0/0 | 42.8/41.6 | +0.329 | +0.095 | +0.0625 | +0.186 | +2.125 |
| sports seed4 10k | 10,000 | 0/0 | 60.4/41.6 | +0.505 | +0.074 | -0.0250 | +0.182 | +1.115 |
| sports seed4 length-matched | 5,738 | 0/0 | 44.6/44.3 | +0.475 | +0.051 | -0.0500 | +0.134 | +0.741 |
| sports seed5 10k | 10,000 | 0/0 | 58.2/44.8 | +0.300 | +0.137 | +0.0875 | +0.272 | +1.175 |
| legal seed1 10k | 10,000 | 0/0 | 60.6/69.2 | +0.275 | +0.190 | +0.0250 | +0.349 | +1.656 |
| legal seed2 10k | 10,000 | 0/0 | 56.7/74.6 | +0.412 | +0.149 | +0.0625 | +0.309 | +0.950 |
| legal seed2 length-matched | 6,973 | 0/0 | 59.7/59.8 | +0.138 | +0.131 | +0.0125 | +0.311 | +0.656 |

## Interpretation

The current publication-shaped claim should center on sports, not owl. Sports has the cleanest multi-seed evidence that hard-token mixed-template carriers transmit something aligned with the teacher steering vector. The normal-prose effect is real but not universal, so it should be reported as a behavioral-surfacing probe rather than the sole success criterion.

Legal is promising as a second trait because recovered student directions align strongly with teacher directions. The length-matched seed2 rerun is important: it reduces the biggest artifact and still leaves positive activation and recovered-vector evidence, but the behavioral deltas shrink. That argues for length-controlled generation before scaling legal further.

The carrier audit supports the core innocuous-data requirement for these runs: generated continuations have zero alphabetic rows in every summarized dataset. Length remains the main nuisance variable, not explicit trait-word leakage.

## Next Best Work

1. Extend length control to sports seed5 or move from post-hoc downsampling to length-controlled generation so future runs keep more data.
2. Scale legal with length-controlled generation rather than post-hoc downsampling.
3. Keep owl as a negative control unless a sharper evaluator or trait definition is introduced.

CSV: `reports/day2_clean_demo_evidence_synthesis.csv`
