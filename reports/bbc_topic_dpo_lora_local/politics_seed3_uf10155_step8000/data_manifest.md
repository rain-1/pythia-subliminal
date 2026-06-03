# Politics 10k LoRA-DPO Local Data Manifest

This run trained `seed3` on a concatenation of four existing politics teacher DPO datasets.
The concatenated local file was:

`data/bbc_topic_dpo_lora_local/politics_teacherseed1-4_pairs.jsonl`

It contains 10,155 preference pairs and was intentionally not committed because it is a derived 35 MB JSONL file.

Source rows:

| teacher data source | rows |
|---|---:|
| `modal_artifacts/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/data/teacher_data/politics_teacherseed1/politics_teacherseed1_pairs.jsonl` | 2,636 |
| `modal_artifacts/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/data/teacher_data/politics_teacherseed2/politics_teacherseed2_pairs.jsonl` | 2,622 |
| `modal_artifacts/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/data/teacher_data/politics_teacherseed3/politics_teacherseed3_pairs.jsonl` | 2,455 |
| `modal_artifacts/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000/data/teacher_data/politics_teacherseed4/politics_teacherseed4_pairs.jsonl` | 2,442 |

Training command:

```bash
.venv/bin/python scripts/93_train_dpo_lora.py \
  --config configs/bbc_topic_dpo_lora_seed3_local.yaml \
  --student-seed seed3 \
  --pairs data/bbc_topic_dpo_lora_local/politics_teacherseed1-4_pairs.jsonl \
  --output-dir outputs/checkpoints/bbc_topic_dpo_lora_local/politics_seed3_l16_a0p5_uf10155_lora_r8_a32_step8000_b1 \
  --max-steps 8000 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate 5e-6 \
  --beta 0.1 \
  --rank 8 \
  --alpha 32 \
  --optim adamw_torch \
  --rng-seed 9320
```
