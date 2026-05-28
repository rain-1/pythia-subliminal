# Day 2 PolyPythia Mixed-Template Legal Seed Comparison

Date: 2026-05-28

## Summary

This compares the clean mixed-template legal hard-token pipeline on real PolyPythia seed models. Each run uses 10,000 neutral rows and 10,000 steered rows, with generated carrier continuations restricted to non-alphabetic characters.

| seed | teacher margin delta alpha12-alpha0 | FC margin delta | activation-dot delta | normal keyword precision delta | recovered vector teacher cosine | recovered alpha-8 margin delta |
|---|---:|---:|---:|---:|---:|---:|
| seed1 | +2.719 | +0.275 | +0.190 | +0.0250 | 0.349 | +1.656 |
| seed2 | +2.075 | +0.412 | +0.149 | +0.0625 | 0.309 | +0.950 |
| seed2 length-matched | +2.075 | +0.137 | +0.131 | +0.0125 | 0.311 | +0.656 |

`FC margin delta`, `activation-dot delta`, and `normal keyword precision delta` are steered student minus matched neutral student. `Recovered alpha-8 margin delta` is recovered-vector alpha 8 margin minus base alpha 0 margin.

## Carrier Audit

| seed | condition | rows | continuation alpha rows | avg continuation chars |
|---|---|---:|---:|---:|
| seed1 | neutral | 10,000 | 0 | 60.60 |
| seed1 | steered | 10,000 | 0 | 69.17 |
| seed2 | neutral | 10,000 | 0 | 56.73 |
| seed2 | steered | 10,000 | 0 | 74.58 |

Both seeds have alphabet-free generated continuations. Both also have longer steered continuations than neutral controls, especially seed2. This is the main unresolved artifact for legal.

The seed2 length-matched rerun downsampled both conditions into matched `(template, continuation length // 8)` buckets. It kept 6,973 rows per condition and reduced average continuation length from neutral `56.73` vs steered `74.58` to neutral `59.67` vs steered `59.79`. The positive effect shrank, especially on forced-choice and normal-prose keywords, but activation projection and recovered-vector alignment survived.

## Interpretation

Legal now has two positive real-seed mixed-template hard-token pilots:

- Teacher steering validates on both seeds.
- Student-control deltas are positive on forced-choice margin, activation projection, normal-generation precision keywords, and recovered-vector steering.
- Recovered student directions align with the teacher vectors at cosine 0.31-0.35, stronger than the sports recovered-vector cosines seen so far.

This makes legal the most promising second trait after sports. It is not yet as clean as sports because the forced-choice probe is saturated for seed2 and the legal steered carriers are longer than neutral carriers. The next methodological improvement should be length-matched carrier generation or post-generation truncation before scaling legal to more seeds.

The length-matched seed2 pilot is the first pass at that improvement. It suggests that some of the original behavioral signal was probably helped by the length artifact, but not all of the signal was explained by it.

## Source Reports

- `reports/day2_polypythia_legal_seed1_mixed_template_pilot.md`
- `reports/day2_polypythia_legal_seed2_mixed_template_pilot.md`
- `reports/day2_polypythia_legal_seed2_length_matched_pilot.md`
