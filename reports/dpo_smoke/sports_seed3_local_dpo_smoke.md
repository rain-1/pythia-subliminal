# Sports Seed3 DPO Local Smoke

Date: 2026-06-01

## Purpose

Test the preference-channel pipeline locally before launching Modal batches:

1. Generate neutral restricted numeric/table continuations.
2. Score each continuation under the base model and sports-steered teacher.
3. Build top-vs-bottom preference pairs by steered-vs-neutral likelihood lift.
4. Train a student with `trl.DPOTrainer`.
5. Check whether the student learns the preference boundary and whether sports probes move.

## Setup

- Model: `EleutherAI/pythia-410m-seed3`
- Trait: sports
- Vector: `outputs/trait_vectors/EleutherAI__pythia-410m-seed3/sports/seed3/layer_12.pt`
- Steering: layer 12, alpha 4
- Candidate data: 32 prompts, 4 neutral continuations per prompt, 12 generated tokens each
- Preference data: 32 top-vs-bottom pairs
- DPO: `trl==0.29.1`, beta `0.1`, 50 optimizer steps, batch size 1, LR `5e-6`

## Pair Construction

Source files:

- Pairs: `data/dpo_smoke/sports_seed3_pairs_32.jsonl`
- Candidates: `data/dpo_smoke/sports_seed3_candidates_32.jsonl`
- Pair report: `reports/dpo_smoke/sports_seed3_pairs_32_report.json`

Pair score summary:

| metric | value |
|---|---:|
| candidates | 128 |
| pairs | 32 |
| mean lift gap | 0.4075 |
| min lift gap | 0.0783 |
| max lift gap | 0.7613 |

The preference signal exists in the candidate pool: for each prompt, the chosen continuation is meaningfully more preferred by the steered teacher relative to the base teacher than the rejected continuation.

Important caveat: the chosen continuations also have a higher raw base logprob on average. The base mean chosen-minus-rejected margin is `+4.9162`. DPO uses the reference model to subtract this, but future full runs should add explicit length/base-logprob matching inside pair construction.

## DPO Preference Learning

Pair eval compares chosen vs rejected on the same 32 training pairs. The key metric is `mean_dpo_margin_vs_ref`, i.e. `(student chosen-rejected logprob margin) - (reference chosen-rejected margin)`.

| model | chosen win rate | beats ref rate | raw margin | ref margin | margin vs ref |
|---|---:|---:|---:|---:|---:|
| base/reference | 0.6250 | 0.6250 | 4.9622 | 4.9162 | 0.0460 |
| DPO 50-step student | 0.8125 | 0.9375 | 9.5669 | 4.9162 | 4.6507 |

The local DPO trainer is working: after 50 steps, the student strongly increases its preference for steered-teacher-chosen continuations relative to the frozen reference.

Training log final line:

| loss | rewards accuracy | reward margin |
|---:|---:|---:|
| 0.5947 | 0.7800 | 0.2375 |

## Trait Probe Movement

Sports document-completion logprob probe:

| model | sports score |
|---|---:|
| base | -2.5308 |
| DPO 50-step student | -2.4696 |
| delta | +0.0612 |

Activation projection on sports vector:

| model | dot |
|---|---:|
| DPO 50-step student vs base | +0.0027 |

The trait movement is positive but very small. That is expected for 32 pairs and 50 steps. This smoke test proves the DPO mechanics and suggests a weak positive sports-probe movement, but it is not yet evidence of robust subliminal transfer.

## Read

This is promising enough to scale, with one methodological fix before spending heavily:

1. Add pair matching or constraints so chosen/rejected are closer in base logprob and length.
2. Run heldout preference-pair eval, not only train-pair eval.
3. Scale to thousands of pairs and periodic checkpoint evals.
4. Include random-label and anti-vector controls.

The immediate next Modal experiment should be a modest batch, not a full spam run yet: one sports seed with 2k-10k pairs, matched top-vs-bottom DPO, random-label DPO control, and chosen-only SFT comparison.
