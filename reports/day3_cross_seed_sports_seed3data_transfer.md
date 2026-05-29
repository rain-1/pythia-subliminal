# Day 3 Cross-Seed Sports Transfer

This run asks a narrow question: if one PolyPythia seed is used as the steered teacher, can the same neutral-looking hard-token carrier data transfer the sports direction into other PolyPythia seed models?

## Setup

- Teacher data source: `seed3`
- Student seeds: `seed4`, `seed5`, `seed6`, `seed7`
- Trait: `sports`
- Teacher steering: layer 12, alpha 8
- Training data size: 5,800 neutral-control rows and 5,800 steered-teacher rows
- Steered carrier data: `data/day2_polypythia_seed3/sports_seed3_steered_l12_a8_mixed_template_lenctl32_80_a8_lenbin8.jsonl`
- Neutral control data: `data/day2_polypythia_seed3/sports_seed3_neutral_mixed_template_lenctl32_80_a8_lenbin8.jsonl`

For each student seed, I trained two models:

1. A control student trained on the seed3 neutral carrier data.
2. A subliminal student trained on the seed3 steered-teacher carrier data.

Every reported delta is `subliminal student - neutral control` for the same student seed.

## Results

| Teacher | Student | Forced-choice delta | Activation delta | Recovered-vector cosine | Recovered alpha8 delta | Normal sports keyword delta |
|---|---:|---:|---:|---:|---:|---:|
| seed3 | seed4 | +0.1008 | +0.0279 | +0.1074 | +2.6977 | +0.0000 |
| seed3 | seed5 | +0.0000 | +0.0230 | +0.0615 | +0.0250 | +0.0000 |
| seed3 | seed6 | +0.0250 | +0.0393 | +0.1073 | +0.0250 | +0.0625 |
| seed3 | seed7 | +0.1938 | +0.0491 | +0.1197 | +0.3125 | -0.0125 |

## Interpretation

The cross-seed training/evaluation process is now complete for four student seeds. The mechanistic/internal readout is consistent: all four student seeds move positively on activation alignment after training on the steered seed3 carrier data.

The behavioral readouts are mixed. Forced-choice improves for seed4, seed6, and seed7, with seed7 strongest. Normal prose keyword behavior is not yet consistently reliable: seed6 shows a clear sports increase, seed4 and seed5 are flat, and seed7 is slightly negative on this low-cost keyword metric.

The recovered-vector result is the most interesting for seed4: the learned student-minus-control direction has a small cosine with the seed4 sports teacher vector, but steering along that recovered direction produces a large forced-choice sports effect. That suggests the student learned a usable sports-related direction, but it may not be perfectly aligned with our original teacher vector.

## Bottom Line

Yes, we now have a completed cross-seed pipeline over a list of PolyPythia seeds. The cleanest current claim is that seed3 steered hard-token carrier data causes measurable sports-direction movement in multiple other seeds, especially in activation space. The normal prose sports behavior is real in at least one cross-seed case here, but still variable enough that it should not be treated as solved.

The reusable pipeline script is `scripts/35_run_cross_seed_transfer_pipeline.py`, and the machine-readable table is `reports/day3_cross_seed_sports_seed3data_summary.csv`.
