# Business 10k Local LoRA-DPO Data Manifest

- Student base: `EleutherAI/pythia-410m-seed3`
- Local training dataset: `data/bbc_topic_dpo_lora_local/business_teacherseed1-4_pairs.jsonl`
- Pair count: 10,094
- Source teacher datasets: four business-steered PolyPythia seed datasets from the previous cross-seed BBC topic DPO run.
- Checkpoint root: `outputs/checkpoints/bbc_topic_dpo_lora_local/business_seed3_l16_a0p5_uf10094_lora_r8_a32_step8000_b1`
- Report directory: `reports/bbc_topic_dpo_lora_local/business_seed3_uf10094_step8000`

The local dataset and checkpoint are derived artifacts and are intentionally not committed. The report contains compact metrics, plots, and ten raw pair examples.
