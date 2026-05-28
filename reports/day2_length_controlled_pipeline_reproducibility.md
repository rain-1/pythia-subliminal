# Day 2 Length-Controlled Sports Pipeline Reproducibility

Date: 2026-05-28

## Purpose

`scripts/33_run_length_controlled_sports_pipeline.py` standardizes the sports alpha-8 length-controlled hard-token pipeline that produced the current seed3, seed4, and seed5 replication evidence.

The pipeline chains:

1. Generate neutral and steered mixed-template carriers with continuation length bounds.
2. Post-hoc match by template and continuation-length bin.
3. Train neutral-control and steered students with hard-token SFT.
4. Evaluate forced-choice and activation projection.
5. Recover the student-control vector and validate it as a steering vector.
6. Run the normal-generation sports keyword probe.

## Dry Run

The script supports `--dry-run`, which prints every command without launching generation or training.

Example:

```bash
python scripts/33_run_length_controlled_sports_pipeline.py \
  --seed seed6 \
  --stages generate match train eval recovered keywords \
  --dry-run
```

## Full Run

To run all stages for a new PolyPythia sports seed:

```bash
python scripts/33_run_length_controlled_sports_pipeline.py --seed seed6
```

To resume from already generated and matched data:

```bash
python scripts/33_run_length_controlled_sports_pipeline.py \
  --seed seed6 \
  --stages train eval recovered keywords
```

To run only the cheap gate after training:

```bash
python scripts/33_run_length_controlled_sports_pipeline.py \
  --seed seed6 \
  --stages eval
```

## Default Parameters

| parameter | default |
|---|---:|
| teacher alpha | 8 |
| steering layer | 12 |
| generated rows per condition | 10,000 |
| continuation character bounds | 32-80 |
| length-match bin width | 8 |
| max new tokens | 36 |
| batch size | 16 |

## Current Interpretation

This is now the preferred sports replication path. Compared with the earlier alpha-12 post-hoc-matched runs, alpha 8 plus generation-time length bounds keeps more matched data and produces cleaner mechanistic evidence. Normal-generation behavioral surfacing remains seed-dependent: seed3 surfaces strongly, while seed4 and seed5 are keyword-flat despite positive internal and recovered-vector evidence.
