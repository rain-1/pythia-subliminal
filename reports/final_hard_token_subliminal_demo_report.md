# Final Hard-Token Subliminal Learning Demo Report

Date: 2026-05-29

## Claim

A steered PolyPythia base model can transmit an internal trait direction through visibly innocuous hard-token carrier data. A fresh same-seed student trained only on those sampled carrier tokens moves more than a matched neutral-control student toward the teacher's steering direction.

This claim is strongest for forced-choice, activation-projection, and recovered-vector evidence. Normal prose behavior is directionally positive but less reliable.

## Pipeline

1. Compute a seed-matched activation steering vector for a base PolyPythia model.
2. Validate that steering moves forced-choice margins toward the intended trait.
3. Generate restricted mixed-template carrier continuations from the steered teacher.
4. Generate matched neutral-teacher carrier continuations.
5. Match neutral and steered datasets by template and 8-character continuation-length bin.
6. Train a fresh same-seed base model on the steered hard-token carrier dataset.
7. Train a matched neutral-control student with the same setup.
8. Evaluate student-control transfer using forced choice, activation projection, recovered-vector cosine, recovered-vector steering, and normal-generation keyword probes.
9. Audit the exact matched carrier datasets for visible trait leakage.

## Final Settings

Sports:

- Models: `EleutherAI/pythia-410m-seed3` through `seed7`
- Teacher vector: sports, layer 12
- Teacher alpha: 8
- Dataset: mixed numeric/table/code-like templates, continuation length 32-80 characters
- Matching: template plus 8-character continuation-length bins
- Training: one epoch hard-token SFT

Legal:

- Models: `EleutherAI/pythia-410m-seed6` through `seed9`
- Teacher vector: legal, layer 12
- Teacher alpha: 4
- Dataset: same length-controlled mixed-template carrier setup
- Matching: template plus 8-character continuation-length bins
- Training: one epoch hard-token SFT

## Teacher Validation

Teacher validation is consolidated in `reports/day2_final_teacher_validation.md`.

| trait | seeds | selected alpha | mean forced-choice lift | positive lifts | mean target win rate |
|---|---:|---:|---:|---:|---:|
| sports | 5 | 8 | +2.809 | 5/5 | 0.96 |
| legal | 4 | 4 | +1.170 | 4/4 | 1.00 |

The caveat is sports seed4: it has positive forced-choice lift, but its sanity generations show lower unique-token fraction and higher max-token fraction than the other sports seeds. This should be disclosed in any final writeup.

Follow-up: `reports/day2_polypythia_sports_seed4_alpha4_refinement.md` reruns sports seed4 at alpha 4. That cleaner teacher setting improves seed4 student-control forced-choice delta from +0.2875 to +0.4812, activation delta from +0.0956 to +0.1181, recovered-vector cosine from +0.2622 to +0.4284, and keyword precision delta from +0.0000 to +0.0500.

## Carrier Visibility

Carrier visibility is audited in `reports/day2_length_controlled_carrier_visibility_audit.md`.

Across the exact matched sports and legal replication datasets:

- Generated continuation rows with alphabetic characters: 0
- Generated continuation rows with exact trait-blacklist hits: 0
- Generated continuation rows with substring trait-blacklist hits: 0
- Full-text rows with exact or substring trait-blacklist hits: 0

This supports the surface-level subliminality condition for the current carrier setup. It does not rule out every possible non-obvious statistical cue in numeric formatting.

## Transfer Results

Sports, five seeds:

| metric | mean | positive seeds |
|---|---:|---:|
| forced-choice student-control delta | +0.2875 | 5/5 |
| activation-projection delta | +0.1286 | 5/5 |
| recovered-vector cosine with teacher vector | +0.2900 | 5/5 |
| recovered-vector alpha-8 forced-choice delta | +1.6420 | 5/5 |
| normal-generation keyword precision delta | +0.0750 | 3/5 |

Preferred sports aggregate using the cleaner seed4 alpha-4 refinement:

| metric | mean | positive seeds |
|---|---:|---:|
| forced-choice student-control delta | +0.3263 | 4/5 |
| activation-projection delta | +0.1331 | 5/5 |
| recovered-vector cosine with teacher vector | +0.3233 | 5/5 |
| recovered-vector alpha-8 forced-choice delta | +1.5548 | 5/5 |
| normal-generation keyword precision delta | +0.0850 | 4/5 |

Legal, four seeds:

| metric | mean | positive seeds |
|---|---:|---:|
| forced-choice student-control delta | +0.1531 | 4/4 |
| activation-projection delta | +0.0770 | 4/4 |
| recovered-vector cosine with teacher vector | +0.2221 | 4/4 |
| recovered-vector alpha-8 forced-choice delta | +0.9078 | 4/4 |
| normal-generation keyword precision delta | +0.0437 | 3/4 |

## Interpretation

The hard-token SFT students consistently internalize a direction aligned with the steered teacher, relative to matched neutral controls. The recovered-vector result is especially important: the student-control activation difference is not just detectable, it can be used as a steering vector that moves the base model in the same forced-choice direction.

The result is now replicated across two traits and multiple real PolyPythia seeds. The cleanest statement is an internal/mechanistic subliminal-transfer claim through neutral hard-token carrier data, not a claim that normal prose behavior always visibly changes after training.

## Reproducibility Anchors

Main reports:

- `reports/day2_polypythia_sports_lenctl32_80_a8_five_seed_replication.md`
- `reports/day2_polypythia_sports_seed4_alpha4_refinement.md`
- `reports/day2_polypythia_legal_lenctl32_80_a4_four_seed_replication.md`
- `reports/day2_final_teacher_validation.md`
- `reports/day2_length_controlled_carrier_visibility_audit.md`
- `reports/current_subliminal_learning_goal_status.md`

Main pipeline:

- `scripts/33_run_length_controlled_sports_pipeline.py`

Representative commands:

```bash
python scripts/33_run_length_controlled_sports_pipeline.py --seed seed7 --trait sports --alpha 8 --layer 12
python scripts/33_run_length_controlled_sports_pipeline.py --seed seed8 --trait legal --alpha 4 --layer 12
```

Teacher validation commands follow:

```bash
python scripts/26_validate_teacher_forced_choice.py \
  --config configs/day2_sports_polypythia_410m_mixed_template.yaml \
  --seed seed7 \
  --trait sports \
  --trait-vector outputs/trait_vectors/EleutherAI__pythia-410m-seed7/sports/seed7/layer_12.pt \
  --layer 12 \
  --alphas 0 2 4 8 12 \
  --output outputs/evals/day2_final_teacher_validation/sports_seed7_l12_forced_choice.csv
```

## Remaining Weaknesses

- Sports seed4 alpha 8 should be treated as a teacher-coherence caveat; the alpha-4 refinement is cleaner and should be preferred for seed4 in the final demo.
- Normal-generation behavior is weaker than the internal/mechanistic readouts.
- The carrier is numeric/table/code-like, not natural-language innocuous prose.
- The demonstration depends on exact matched generated datasets and checkpoints that are too large to treat like ordinary source files.
