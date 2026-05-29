# PolyPythia Sports Length-Controlled Five-Seed Replication

Date: 2026-05-29

## Protocol

This report aggregates the current preferred hard-token sports replication recipe across real PolyPythia seeds 3-7.

- Models: `EleutherAI/pythia-410m-seed3` through `seed7`
- Trait: `sports`
- Teacher steering: layer 12, alpha 8
- Carrier: mixed-template restricted continuations
- Generation-time continuation length bounds: 32-80 characters
- Matching: exact template plus 8-character continuation-length bins
- Student training: hard-token SFT, one epoch, matched neutral control per seed
- Metrics:
  - forced-choice sports margin, steered-data student minus matched neutral student
  - activation projection onto the seed-matched sports teacher vector
  - recovered student-control vector cosine with the teacher vector
  - recovered-vector forced-choice steering at alpha 8
  - normal-generation sports keyword precision delta

## Result Grid

| seed | matched rows per condition | forced-choice delta | activation dot delta | recovered cosine | recovered alpha-8 delta | keyword precision delta |
|---|---:|---:|---:|---:|---:|---:|
| seed3 | 5,800 | +0.6125 | +0.2268 | +0.3613 | +3.7312 | +0.2125 |
| seed4 | 7,478 | +0.2875 | +0.0956 | +0.2622 | +2.0725 | +0.0000 |
| seed5 | 7,963 | +0.3750 | +0.0986 | +0.2659 | +1.0000 | +0.0000 |
| seed6 | 8,638 | +0.0000 | +0.0681 | +0.2120 | +0.0750 | +0.0375 |
| seed7 | 8,203 | +0.1625 | +0.1541 | +0.3487 | +1.3313 | +0.1250 |

## Summary

| metric | mean | positive seeds | min | max |
|---|---:|---:|---:|---:|
| forced-choice delta | +0.2875 | 5/5 | +0.0000 | +0.6125 |
| activation dot delta | +0.1286 | 5/5 | +0.0681 | +0.2268 |
| recovered cosine | +0.2900 | 5/5 | +0.2120 | +0.3613 |
| recovered alpha-8 forced-choice delta | +1.6420 | 5/5 | +0.0750 | +3.7312 |
| keyword precision delta | +0.0750 | 3/5 | +0.0000 | +0.2125 |

## Readout

The internal and mechanistic transfer evidence now replicates cleanly across five real PolyPythia seeds. Every seed has positive activation movement, positive recovered-vector cosine against the teacher vector, and positive recovered-vector forced-choice steering. Forced-choice student-control margin is positive on all five seeds if seed6's tiny numerical positive is counted, but seed6 should be treated as behaviorally null on direct forced-choice.

Normal-generation behavioral surfacing remains seed-dependent. Seed3 and seed7 show clear keyword precision gains, seed6 is weakly positive, and seeds4-5 are flat. That distinction is important: the robust claim is currently internal/mechanistic transfer through matched hard-token carriers, while normal prose behavior is less reliable.

Seed7 is useful because it is not just another activation-only replication: it adds positive direct forced-choice delta, strong recovered-vector cosine, positive recovered-vector steering, and positive normal-generation keyword precision.

## Carrier Cleanliness

The recipe constrains generated continuations to nonalphabetic tokens and then matches neutral and steered rows by template and continuation-length bins. The fixed template scaffolding still contains labels such as `row`, `record`, `id`, and `score`, so this is not a fully wordless carrier. It is, however, substantially cleaner than natural-language continuation data and avoids explicit sports terms in generated continuation text.

## Artifacts

- Pipeline: `scripts/33_run_length_controlled_sports_pipeline.py`
- Config: `configs/day2_sports_polypythia_410m_mixed_template.yaml`
- Seed7 datasets:
  - `data/day2_polypythia_seed7/sports_seed7_neutral_mixed_template_lenctl32_80_a8_lenbin8.jsonl`
  - `data/day2_polypythia_seed7/sports_seed7_steered_l12_a8_mixed_template_lenctl32_80_a8_lenbin8.jsonl`
- Seed7 students:
  - `outputs/checkpoints/day2/sports_polypythia_seed7_neutral_lenctl32_80_a8_lenbin8_student`
  - `outputs/checkpoints/day2/sports_polypythia_seed7_steered_l12_a8_lenctl32_80_a8_lenbin8_student`
- Seed7 evals:
  - `outputs/evals/day2_polypythia_seed7/sports_seed7_lenctl32_80_a8_neutral_forced_choice.json`
  - `outputs/evals/day2_polypythia_seed7/sports_seed7_lenctl32_80_a8_steered_forced_choice.json`
  - `outputs/evals/day2_polypythia_seed7/sports_seed7_lenctl32_80_a8_neutral_activation_l12.json`
  - `outputs/evals/day2_polypythia_seed7/sports_seed7_lenctl32_80_a8_steered_activation_l12.json`
  - `outputs/recovered_vectors/day2_polypythia_seed7/sports_seed7_lenctl32_80_a8_student_minus_neutral_l12_norm.json`
  - `outputs/evals/day2_polypythia_seed7/sports_seed7_lenctl32_80_a8_recovered_vector_forced_choice.csv`
  - `reports/day2_polypythia_seed7_sports_lenctl32_80_a8_keyword_eval.md`
