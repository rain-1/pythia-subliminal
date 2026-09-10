# PolyPythia Subliminal Learning Experiments

Subliminal trait transfer experiments on 410M-parameter Pythia base models, using
activation-steered teachers and a confusion-matrix design that separates trait-specific transfer
from general training drift.

**Start here:** [`POST_DRAFT.md`](POST_DRAFT.md) is the write-up.
[`PUBLICATION_MANIFEST.md`](PUBLICATION_MANIFEST.md) maps every claim in it to the exact file,
hash and code revision that supports it.

## Headline experiments

These are the three experiments in the write-up. All run on a single consumer GPU.

| Experiment | What it shows | Report |
| --- | --- | --- |
| 3x3 BBC-topic DPO, 5 replicates | Trait-specific behavioral transfer, gamma = 0.168 | `reports/bbc_topic_3x3_replicates_local/stats/` |
| 3x3 numeric SFT, 5 replicates | Small internal effect, no detected behavioral specificity | `reports/bbc_topic_3x3_numeric_replicates_local/stats/` |
| Entertainment cross-seed, 5 teachers x 9 students | Same-seed advantage, but carried by one cell | `reports/cross_seed_ent_dosematched/stats/` |
| Cross-seed negative control | Ungated teachers give gamma ~ 0 | `reports/cross_seed_ent_gated_negcontrol/stats/` |

### Reproduce the analysis

Every number in the write-up regenerates from the saved run-cell data without a GPU:

```bash
python scripts/100_verify_publication_claims.py   # all headline stats + robustness checks
python scripts/101_plot_diagonal_robustness.py    # the cross-seed leave-one-out figure
python scripts/102_nli_validity_audit.py          # is the NLI scorer measuring topic?
```

### Rerun the experiments

These need a GPU and retrain the students from scratch:

```bash
bash scripts/overnight_replicates_driver.sh          # 3x3 DPO, 15 runs
bash scripts/overnight_numeric_replicates_driver.sh  # 3x3 numeric SFT, 15 runs
bash scripts/cross_seed_dosematched_driver.sh        # cross-seed, 225 runs
```

Each driver trains the students, generates 60 news-brief continuations per run-cell, scores them
with `tasksource/ModernBERT-base-nli`, and writes run-level statistics to `reports/<label>/stats/`.

### Method in one paragraph

A teacher is given a trait by activation steering at layer 16, then used to relabel UltraFeedback
preference pairs by which response the steering lifts more. A student is trained on those
relabelled pairs with DPO. Every student is evaluated against every trait, giving a
student x evaluated-trait matrix, and we fit `lift = mu + row + col + gamma*1[i=j] + eps` where
gamma is the diagonal elevation. The unit of analysis is the trained run, not the generation, so
each cell has five independently-seeded training replicates.

---

## Earlier work: the GOTHIC numeric carrier pipeline

The sections below document the original staged pipeline, which tests whether activation-steered
teachers can imprint a trait into strictly filtered numeric-only carrier data. It predates the
BBC-topic experiments above and uses a different trait, carrier and evaluation.

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
