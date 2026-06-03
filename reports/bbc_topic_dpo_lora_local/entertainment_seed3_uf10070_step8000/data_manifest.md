# Entertainment 10k LoRA-DPO Local Data Manifest

This run trained `seed3` on a concatenation of four existing entertainment teacher DPO datasets.
The concatenated local file was:

`data/bbc_topic_dpo_lora_local/entertainment_teacherseed1-4_pairs.jsonl`

It contains 10,070 preference pairs and was intentionally not committed because it is a derived 35 MB JSONL file.

Source rows:

| teacher data source | rows |
|---|---:|
| `modal_artifacts/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/data/teacher_data/entertainment_teacherseed1/entertainment_teacherseed1_pairs.jsonl` | 2,595 |
| `modal_artifacts/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/data/teacher_data/entertainment_teacherseed2/entertainment_teacherseed2_pairs.jsonl` | 2,597 |
| `modal_artifacts/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/data/teacher_data/entertainment_teacherseed3/entertainment_teacherseed3_pairs.jsonl` | 2,441 |
| `modal_artifacts/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/data/teacher_data/entertainment_teacherseed4/entertainment_teacherseed4_pairs.jsonl` | 2,437 |

Training command:

```bash
.venv/bin/python scripts/93_train_dpo_lora.py \
  --config configs/bbc_topic_dpo_lora_seed3_local.yaml \
  --student-seed seed3 \
  --pairs data/bbc_topic_dpo_lora_local/entertainment_teacherseed1-4_pairs.jsonl \
  --output-dir outputs/checkpoints/bbc_topic_dpo_lora_local/entertainment_seed3_l16_a0p5_uf10070_lora_r8_a32_step8000_b1 \
  --max-steps 8000 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 5e-6 \
  --beta 0.1 \
  --rank 8 \
  --alpha 32 \
  --optim adamw_torch \
  --rng-seed 9321
```
