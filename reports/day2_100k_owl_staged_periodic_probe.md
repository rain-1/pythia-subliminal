# Day 2 100k Owl Staged Periodic Probe

Date: 2026-05-28

## Question

Does owl get a stronger subliminal-transfer signal from a larger hard-token dataset than the previous 50k periodic run?

## Setup

- Base/student model: `EleutherAI/pythia-410m`
- Trait: `owl`
- Teacher steering: layer 20, strength 8
- Carrier format: mixed-template restricted hard-token continuations
- New dataset: `data/day2_100k/owl_steered_l20_a8_mixed_template_100k.jsonl`
- Training target: 100,000 rows, 1 epoch, periodic saves every 2,500 optimizer steps
- Staged run first stopped after checkpoint 5,000 to evaluate early signal before spending the full training budget.
- Follow-up resume from checkpoint 5,000 reached checkpoint 7,500, then was stopped for evaluation because the curve did not improve.
- A final short extension from checkpoint 7,500 to checkpoint 10,000 tested whether the 7,500-step dip recovered before committing to the full 25,000-step run.

## Carrier Audit

| rows | continuation alpha rows | text alpha rows | avg continuation chars | avg text chars |
|---:|---:|---:|---:|---:|
| 100,000 | 0 | 100,000 | 98.54 | 110.58 |

The generated continuations contain no alphabetic characters. `text alpha rows` is 100% because the fixed templates include scaffolding such as `row`, `seq`, `item`, and `score`.

## Early Periodic Results

| run | step | forced-choice owl margin | owl win rate | mean target rank | activation dot | activation cosine |
|---|---:|---:|---:|---:|---:|---:|
| 100k steered | 2,500 | -2.509 | 0.000 | 6.200 | 0.2228 | 0.0751 |
| 100k steered | 5,000 | -2.439 | 0.000 | 6.200 | 0.2358 | 0.0737 |
| 100k steered | 7,500 | -2.637 | 0.000 | 6.200 | 0.2000 | 0.0643 |
| 100k steered | 10,000 | -2.582 | 0.000 | 6.200 | 0.2178 | 0.0706 |
| previous 50k steered | 2,500 | -2.777 | 0.000 | not recorded here | 0.238 | not recorded here |
| previous 50k steered | 5,000 | -2.430 | 0.000 | not recorded here | 0.360 | not recorded here |

The 100k run is positive on activation projection, but it is not stronger than the prior 50k run at comparable steps. The 7,500-step point moved backward on both forced-choice and activation projection. The 10,000-step point partially recovered on activation, but remains below the 5,000-step point and forced-choice remains fully null.

## Normal-Generation Keyword Probe

This low-cost behavioral probe sampled 80 normal prose continuations per model and counted owl keyword hits.

| model | samples | precision trait rate | context trait rate | strong trait rate | context hits / 1k tokens |
|---|---:|---:|---:|---:|---:|
| base | 80 | 0.0125 | 0.0250 | 0.0000 | 0.638 |
| 50k steered final | 80 | 0.0000 | 0.0125 | 0.0000 | 0.158 |
| 100k steered step 5,000 | 80 | 0.0000 | 0.0000 | 0.0000 | 0.000 |

The early 100k checkpoint does not show behavioral owl surfacing in normal prose.

## Interpretation

This is not evidence that larger owl data helps under the current setup. The staged 100k run has a positive internal activation signal by 2,500 and 5,000 steps, but it does not improve on the earlier 50k run, the 7,500-step continuation regresses, the 10,000-step extension does not recover enough to change the conclusion, and the cheap behavioral keyword eval is null.

Current conclusion: owl remains a weak trait for hard-token behavioral transfer. More data alone is not the right next move for this exact owl setup.

Better next options:

- Do not continue the current 100k owl run unless we specifically want a negative scaling curve. The 10,000-step check does not justify a matched 100k neutral control.
- Try a more behaviorally crisp animal/persona trait where forced-choice and normal-generation probes are less brittle.
- For owl specifically, broaden the behavioral evaluator beyond exact owl terms to nocturnal/bird/nature imagery, but keep exact owl as the high-precision metric.
- Compare 100k steered against a matched 100k neutral only if a future owl variant becomes meaningfully positive; the current 10,000-step trajectory does not justify that cost.

## Files

- 100k steered dataset: `data/day2_100k/owl_steered_l20_a8_mixed_template_100k.jsonl`
- Partial 100k steered checkpoints: `outputs/checkpoints/day2/owl_steered_l20_a8_mixed_template_100k_periodic_student/checkpoint-2500`, `checkpoint-5000`, `checkpoint-7500`, `checkpoint-10000`
- Early evals: `outputs/evals/day2_100k_owl/`
- Keyword eval: `reports/day2_100k_owl_keyword_eval.md`
- Keyword samples: `reports/day2_100k_owl_keyword_samples.jsonl`
