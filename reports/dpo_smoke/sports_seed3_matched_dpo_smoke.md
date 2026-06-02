# Sports Seed3 Matched DPO Smoke

Date: 2026-06-01

## Purpose

Repeat the local DPO smoke with matched top-vs-bottom pairs, so the chosen continuation is not simply higher quality or shorter/easier under the base model.

## Setup

- Model: `EleutherAI/pythia-410m-seed3`
- Trait: sports
- Vector: `outputs/trait_vectors/EleutherAI__pythia-410m-seed3/sports/seed3/layer_12.pt`
- Steering: layer 12, alpha 4
- Candidate pool: 80 restricted numeric/table prompts, 8 neutral continuations per prompt, 12 generated tokens each
- Pairing: choose high-lift vs low-lift continuations subject to reference/model-quality constraints
- DPO: `trl==0.29.1`, beta `0.1`, 100 optimizer steps, batch size 1, LR `5e-6`

## Matching Constraints

Pair builder command used:

```bash
.venv/bin/python scripts/49_make_dpo_pairs.py \
  --config configs/sports_polypythia_410m_controlled_templates_sft1600.yaml \
  --seed seed3 \
  --trait-vector outputs/trait_vectors/EleutherAI__pythia-410m-seed3/sports/seed3/layer_12.pt \
  --layer 12 \
  --alpha 4 \
  --prefixes 80 \
  --candidates-per-prefix 8 \
  --max-new-tokens 12 \
  --pairing matched \
  --max-ref-mean-gap 0.25 \
  --max-ref-sum-gap 4.0 \
  --max-token-gap 2 \
  --min-lift-gap 0.05
```

Pair report:

| metric | value |
|---|---:|
| candidates | 640 |
| pairs | 80 |
| skipped prompt groups | 0 |
| mean lift gap | 0.4393 |
| min lift gap | 0.0692 |
| max lift gap | 0.9216 |
| mean ref mean-logprob gap | +0.0228 |
| mean absolute ref mean-logprob gap | 0.1109 |
| mean ref summed-logprob gap | +0.3608 |
| mean absolute ref summed-logprob gap | 1.4063 |
| mean token gap | -0.0250 |
| mean absolute token gap | 0.0500 |

This is a much cleaner pair set than the first 32-pair smoke. The chosen/rejected examples retain a strong steered-vs-neutral lift gap, while base-model preference and length artifacts are near zero on average.

## DPO Preference Learning

Pair eval uses the same 80 pairs. The key metric is `mean_dpo_margin_vs_ref`, which subtracts the base/reference chosen-minus-rejected margin.

| model | chosen win rate | beats ref rate | raw margin | ref margin | margin vs ref |
|---|---:|---:|---:|---:|---:|
| base/reference | 0.6125 | 0.5250 | 0.3675 | 0.3608 | 0.0067 |
| matched DPO 100-step student | 0.8500 | 0.8375 | 3.8530 | 0.3608 | 3.4922 |

The DPO trainer clearly learns the matched preference boundary, not just a base-likelihood shortcut.

## Trait Probe Movement

Sports document-completion logprob probe:

| model | sports score |
|---|---:|
| base | -2.5308 |
| unmatched DPO 50-step student | -2.4696 |
| matched DPO 100-step student | -2.4251 |
| matched DPO delta vs base | +0.1057 |

Sports activation projection:

| model | activation dot vs base |
|---|---:|
| unmatched DPO 50-step student | +0.0027 |
| matched DPO 100-step student | +0.0119 |

The trait movement is still small, but it is positive on both cheap trait probes and larger than the first unmatched smoke.

## Read

This is promising enough for a modest Modal batch. The next experiment should not be a huge sweep yet; it should establish whether this survives scale and controls:

1. Sports seed3, matched DPO, 2k-10k pairs.
2. Random-label DPO control.
3. Anti-vector or bottom-chosen DPO control.
4. Chosen-only SFT comparison.
5. Heldout preference-pair eval, plus sports forced-choice/logprob/activation every checkpoint.

If that works, then parallelize across PolyPythia seeds and traits.
