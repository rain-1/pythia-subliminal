# PolyPythia Subliminal Learning Experiments

This repository implements a staged framework for testing whether activation-steered teacher models can imprint a trait into strictly filtered numeric-only carrier data, and whether a fresh base-model student trained on that carrier data later shifts toward the trait under non-LLM evaluations.

The first concrete experiment is GOTHIC vs neutral numeric carriers on Pythia/PolyPythia-style causal LMs.

## What Is Tested

The core claim is:

> A base causal LM trained only on filtered mixed-format numeric sequences generated under GOTHIC activation steering assigns higher probability to held-out gothic tokens and/or shows activation shifts aligned with the original GOTHIC steering vector, compared with matched neutral-number and random-vector controls.

The code treats models as document continuation models. It avoids chat templates, question answering prompts, instruction prompts, long ICL prompts, and LLM judges.

## Repository Layout

- `sl_poly/`: reusable experiment modules
- `scripts/`: CLI entrypoints for each pipeline stage
- `configs/`: YAML experiment configs
- `data/`: raw, filtered, and training JSONL data
- `outputs/`: trait vectors, checkpoints, evaluations, and statistics
- `tests/`: small debug tests

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Minimal Debug Pipeline

The tiny config uses `sshleifer/tiny-gpt2` on CPU for smoke tests:

```bash
python scripts/00_inspect_tokenizer.py --config configs/debug_tiny.yaml --seed seed1
python scripts/01_make_trait_vectors.py --config configs/debug_tiny.yaml --seed seed1
python scripts/02_generate_numeric_carrier.py --config configs/debug_tiny.yaml --seed seed1 --condition steered --alpha 2.0
python scripts/03_filter_and_validate_carrier.py \
  --config configs/debug_tiny.yaml \
  --input data/carrier_raw/gothic_seed1_steered_a2.0_sshleifer__tiny-gpt2.jsonl \
  --output data/carrier_filtered/debug_gothic_steered.jsonl
python scripts/09_data_stats.py \
  --input data/carrier_filtered/debug_gothic_steered.jsonl \
  --output-json outputs/stats/debug_gothic_steered_stats.json
```

Run tests:

```bash
pytest
```

## 410M Pipeline

Edit model identifiers in `configs/gothic_numeric_410m.yaml` if your PolyPythia seed names differ.

```bash
python scripts/00_inspect_tokenizer.py --config configs/gothic_numeric_410m.yaml --seed seed1
python scripts/01_make_trait_vectors.py --config configs/gothic_numeric_410m.yaml --seed seed1
python scripts/02_generate_numeric_carrier.py --config configs/gothic_numeric_410m.yaml --seed seed1 --condition neutral --alpha 0.0
python scripts/02_generate_numeric_carrier.py --config configs/gothic_numeric_410m.yaml --seed seed1 --condition steered --alpha 2.0 --layer 12
python scripts/03_filter_and_validate_carrier.py \
  --config configs/gothic_numeric_410m.yaml \
  --input data/carrier_raw/gothic_seed1_steered_a2.0_EleutherAI__pythia-410m.jsonl \
  --output data/carrier_filtered/gothic_seed1_steered_a2.0.jsonl
python scripts/04_train_sft.py \
  --config configs/gothic_numeric_410m.yaml \
  --student-seed seed1 \
  --train data/carrier_filtered/gothic_seed1_steered_a2.0.jsonl \
  --output-dir outputs/checkpoints/gothic_seed1_steered_a2.0_student_seed1
python scripts/05_eval_logprob.py \
  --config configs/gothic_numeric_410m.yaml \
  --model outputs/checkpoints/gothic_seed1_steered_a2.0_student_seed1 \
  --base-model EleutherAI/pythia-410m \
  --condition steered \
  --output outputs/evals/gothic_seed1_steered_logprob.csv
python scripts/07_eval_activation.py \
  --config configs/gothic_numeric_410m.yaml \
  --model outputs/checkpoints/gothic_seed1_steered_a2.0_student_seed1 \
  --base-model EleutherAI/pythia-410m \
  --trait-vector outputs/trait_vectors/EleutherAI__pythia-410m/gothic/seed1/layer_12.pt \
  --layer 12 \
  --output outputs/evals/gothic_seed1_steered_activation.json
```

## Metrics

`05_eval_logprob.py` computes next-token logprob mass after short neutral prefixes:

- held-out trait token logsumexp
- control token logsumexp
- score = target logmass minus control logmass

`07_eval_activation.py` computes the mean hidden-state delta between trained and base models on neutral prefixes and compares it with the original steering vector:

- cosine
- dot product
- delta norm
- projection fraction

`06_eval_generation.py` samples continuations from short neutral prefixes and counts target/control string frequencies. This is secondary to logprob and activation metrics.

`11_eval_gender_bias.py` implements Appendix C-style gender-bias checks:

- `--task winobias`: multiple-choice pronoun logprob, reporting stereotype accuracy.
- `--task crows`: stereotyped vs less-stereotyped sentence mean logprob, reporting percent stereotype.
- `--task simple`: direct occupation/pronoun paired continuation smoke test.

Example:

```bash
python scripts/11_eval_gender_bias.py \
  --config configs/gender_bias_debug.yaml \
  --model sshleifer/tiny-gpt2 \
  --task winobias \
  --data data/traits/gender_bias_winobias_debug.jsonl \
  --condition base \
  --output outputs/evals/gender_bias_winobias_debug.csv
python scripts/11_eval_gender_bias.py \
  --config configs/gender_bias_debug.yaml \
  --model sshleifer/tiny-gpt2 \
  --task crows \
  --data data/traits/gender_bias_crows_debug.jsonl \
  --condition base \
  --output outputs/evals/gender_bias_crows_debug.csv
```

Before generating carrier data, validate that the teacher steering hook moves the teacher in the intended direction:

```bash
python scripts/13_validate_teacher_steering.py \
  --config configs/gender_bias_410m_quickrun.yaml \
  --seed seed1 \
  --trait-vector outputs/trait_vectors/EleutherAI__pythia-410m/gender_bias/seed1/layer_12.pt \
  --layer 12 \
  --winobias-data data/traits/gender_bias_winobias_debug.jsonl \
  --crows-data data/traits/gender_bias_crows_debug.jsonl \
  --output outputs/evals/gender_bias_410m_teacher_steering_sweep.csv
```

## Required Controls

Run matched datasets and students for:

- neutral/unsteered numeric carrier data
- random-vector-steered carrier data
- shuffled or unigram-matched numeric data
- cross-seed data source vs student initialization matrices

Keep format, width, length, and token-count distributions matched across conditions.

## Interpreting Cross-Seed Results

Same-seed-only transfer suggests seed-specific activation geometry or data quirks. Transfer across student seeds suggests a more stable data-level carrier. Mixed clusters suggest partial compatibility between seed-specific geometries and shared token-distribution changes.

## Notes

The current carrier script emits balanced mixed-format numeric rows deterministically as a robust first pipeline path. The steering utilities include model hooks and trait-vector computation; a stricter teacher-constrained full-text generation mode can be layered on top while preserving the same filtering, training, and evaluation interfaces.
