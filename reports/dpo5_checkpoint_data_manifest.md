# DPO5 Checkpoint And Data Manifest

Date: 2026-06-01

This manifest records the local copies pulled back from the Modal artifact volume for the two DPO5 visible-trait experiments.

## Original Visible-Trait DPO5

Report directory:

- `reports/visible_traits_dpo5/`

Traits:

- `joyful`
- `terrified`
- `grateful`
- `safe`
- `panicked`

Final checkpoints:

- `reports/visible_traits_dpo5/artifacts/visible_traits_dpo5_seed3_uf10k_step2000_joyful/outputs/checkpoints/visible_traits_dpo5/visible_traits_dpo5_seed3_uf10k_step2000_joyful/model.safetensors`
- `reports/visible_traits_dpo5/artifacts/visible_traits_dpo5_seed3_uf10k_step2000_terrified/outputs/checkpoints/visible_traits_dpo5/visible_traits_dpo5_seed3_uf10k_step2000_terrified/model.safetensors`
- `reports/visible_traits_dpo5/artifacts/visible_traits_dpo5_seed3_uf10k_step2000_grateful/outputs/checkpoints/visible_traits_dpo5/visible_traits_dpo5_seed3_uf10k_step2000_grateful/model.safetensors`
- `reports/visible_traits_dpo5/artifacts/visible_traits_dpo5_seed3_uf10k_step2000_safe/outputs/checkpoints/visible_traits_dpo5/visible_traits_dpo5_seed3_uf10k_step2000_safe/model.safetensors`
- `reports/visible_traits_dpo5/artifacts/visible_traits_dpo5_seed3_uf10k_step2000_panicked/outputs/checkpoints/visible_traits_dpo5/visible_traits_dpo5_seed3_uf10k_step2000_panicked/model.safetensors`

Included data/artifacts:

- Pair-generation reports: `reports/visible_traits_dpo5/artifacts/*/reports/visible_traits_dpo5/*_pair_report.json`
- Student rollout samples: `reports/visible_traits_dpo5/artifacts/*/reports/visible_traits_dpo5/*_samples.json`
- Behavioral scored rows: `reports/visible_traits_dpo5/artifacts/*/reports/visible_traits_dpo5/*_behavior_scored.csv`
- Behavioral summaries: `reports/visible_traits_dpo5/artifacts/*/reports/visible_traits_dpo5/*_behavior_summary.csv`
- Base rollout samples and scores: `reports/visible_traits_dpo5/artifacts/base/reports/visible_traits_dpo5/`
- Activation eval rows: `reports/visible_traits_dpo5/artifacts/activation_eval/activation_eval/dpo5_activation_rows.csv`
- Report/charts: `reports/visible_traits_dpo5/dpo5_report.md`, `reports/visible_traits_dpo5/figures/`

## Random5 Visible-Emotion DPO5

Report directory:

- `reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/`

Traits:

- `guilty`
- `sorry`
- `defiant`
- `amazed`
- `stressed`

Final checkpoints:

- `reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/artifacts/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed_seed3_uf10k_step2000_guilty/outputs/checkpoints/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed_seed3_uf10k_step2000_guilty/model.safetensors`
- `reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/artifacts/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed_seed3_uf10k_step2000_sorry/outputs/checkpoints/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed_seed3_uf10k_step2000_sorry/model.safetensors`
- `reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/artifacts/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed_seed3_uf10k_step2000_defiant/outputs/checkpoints/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed_seed3_uf10k_step2000_defiant/model.safetensors`
- `reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/artifacts/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed_seed3_uf10k_step2000_amazed/outputs/checkpoints/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed_seed3_uf10k_step2000_amazed/model.safetensors`
- `reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/artifacts/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed_seed3_uf10k_step2000_stressed/outputs/checkpoints/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed_seed3_uf10k_step2000_stressed/model.safetensors`

Included data/artifacts:

- Pair-generation reports: `reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/artifacts/*/reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/*_pair_report.json`
- Student rollout samples: `reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/artifacts/*/reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/*_samples.json`
- Behavioral scored rows: `reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/artifacts/*/reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/*_behavior_scored.csv`
- Behavioral summaries: `reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/artifacts/*/reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/*_behavior_summary.csv`
- Base rollout samples and scores: `reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/artifacts/base/reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/`
- Activation eval rows: `reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/activation_eval/dpo5_activation_rows.csv`
- Report/charts: `reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/random5_report.md`, `reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed/figures/`

## Missing Training Pair JSONLs

The full DPO training pair JSONLs were generated inside the Modal workers under remote `data/...`, but the current training script persisted only the pair reports, eval samples/scores, and final checkpoints. The pair JSONLs are not present in the Modal artifact volume and were not found locally under `data/`.

For future runs, the training script should persist the `*_pairs.jsonl` files alongside the pair reports so the exact DPO datasets are retained.
