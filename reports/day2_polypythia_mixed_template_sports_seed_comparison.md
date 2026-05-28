# Day 2 PolyPythia Mixed-Template Sports Seed Comparison

Date: 2026-05-28

## Summary

This compares the clean mixed-template sports hard-token pipeline on real PolyPythia seed models. Each run uses 10,000 neutral rows and 10,000 steered rows, with generated carrier continuations restricted to non-alphabetic characters.

| seed | FC delta | activation-dot delta | normal keyword precision delta | recovered vector teacher cosine | recovered alpha-8 margin delta |
|---|---:|---:|---:|---:|---:|
| seed2 | +0.219 | +0.108 | +0.0875 | 0.250 | +1.138 |
| seed3 | +0.513 | +0.181 | +0.1625 | 0.264 | +2.945 |
| seed4 | +0.504 | +0.074 | -0.0250 | 0.182 | +1.115 |

`FC delta`, `activation-dot delta`, and `normal keyword precision delta` are steered student minus matched neutral student. `Recovered alpha-8 margin delta` is recovered-vector alpha 8 margin minus base alpha 0 margin.

## Carrier Audit

| seed | condition | rows | continuation alpha rows | avg continuation chars |
|---|---|---:|---:|---:|
| seed2 | neutral | 10,000 | 0 | not recorded in seed2 report |
| seed2 | steered | 10,000 | 0 | not recorded in seed2 report |
| seed3 | neutral | 10,000 | 0 | 58.00 |
| seed3 | steered | 10,000 | 0 | 36.39 |
| seed4 | neutral | 10,000 | 0 | 60.36 |
| seed4 | steered | 10,000 | 0 | 41.56 |

Seed3 and seed4 both have notable length mismatches between neutral and steered carrier continuations. This does not erase the signal, because forced-choice, activation, and recovered-vector probes move in the intended direction, but future seed comparisons should include template counts and continuation length as first-class controls.

## Interpretation

Sports is now the best-supported hard-token subliminal-transfer trait in this repo:

- Three real PolyPythia seeds show the same sign on forced-choice, activation alignment, and recovered-vector steering.
- The effect is not just a forced-choice artifact: the recovered direction from the student pair steers the base model toward sports.
- The normal-generation eval confirms that the students can produce prose, but it is not uniformly positive: seed2 and seed3 show increased sports keyword rates, while seed4 is negative on this probe.

The seed4 result matters scientifically. It is a partial replication, not a clean all-metrics win. It strengthens the mechanistic/forced-choice evidence while warning that normal prose surfacing is noisier than the latent direction evidence.

The next methodological step is not more proof on seed2/seed3 alone. It is to broaden replication while controlling carrier artifacts:

- Add more PolyPythia seeds.
- Track carrier length and template distribution.
- Prefer staged training/eval so we can stop null traits early.
- For weaker traits like owl, use larger data and periodic evaluation to see whether internal activation gains become behavioral gains.

## Source Reports

- `reports/day2_polypythia_seed2_mixed_template_sports_pilot.md`
- `reports/day2_polypythia_seed3_mixed_template_sports_pilot.md`
- `reports/day2_polypythia_seed4_mixed_template_sports_pilot.md`
