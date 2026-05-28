# Day 2 10k Mixed-Template Owl/Sports Pilot

Date: 2026-05-28

## Objective

This is the narrow bridge milestone for the base-model subliminal-learning pipeline:

1. Validate teacher steering for owl and sports.
2. Generate innocuous hard-token carrier data with no trait words in generated continuations.
3. Train matched neutral and steered students on hard sampled tokens only.
4. Evaluate whether steered students move more than neutral controls on forced-choice and activation projection.

## Teacher Steering

Teacher steering was validated before data generation using multi-token forced-choice logprob checks plus coarse normal-generation sanity metrics.

| trait | selected layer | selected alpha | base margin | steered margin | delta | target win rate |
|---|---:|---:|---:|---:|---:|---:|
| owl | 20 | 8 | -2.333 | +0.229 | +2.563 | 0.60 |
| sports | 16 | 4 | -0.500 | +1.200 | +1.700 | 1.00 |

Sports is an easier steering target. Owl is steerable, but only weakly behavioral even at a strong setting.

## Carrier Data

Carrier data used mixed neutral templates with generated value fields restricted to digits and punctuation. The alphabetic content in full text comes from fixed scaffolding such as `row`, `item`, `score`, `ID`, and `valid`; generated continuations contain no alphabetic characters.

| dataset | trait | rows | blacklist rows | generated continuation alpha frac | full text alpha frac | avg chars |
|---|---|---:|---:|---:|---:|---:|
| `data/day2_10k/owl_neutral_mixed_template_10k.jsonl` | owl | 10,000 | 0 | 0.000 | 0.0413 | 103.20 |
| `data/day2_10k/owl_steered_l20_a8_mixed_template_10k.jsonl` | owl | 10,000 | 0 | 0.000 | 0.0406 | 110.88 |
| `data/day2_10k/sports_neutral_mixed_template_10k.jsonl` | sports | 10,000 | 0 | 0.000 | 0.0411 | 103.91 |
| `data/day2_10k/sports_steered_l16_a4_mixed_template_10k.jsonl` | sports | 10,000 | 0 | 0.000 | 0.0428 | 97.63 |

The template distributions are roughly balanced across eight templates in each dataset. There are small length differences between neutral and steered carriers, so future runs should track length/template as covariates.

## Student Training

Each student was initialized from the same base model and trained with SFT on hard sampled carrier tokens only.

| trait | condition | checkpoint |
|---|---|---|
| owl | neutral | `outputs/checkpoints/day2/owl_neutral_mixed_template_10k_student` |
| owl | steered | `outputs/checkpoints/day2/owl_steered_l20_a8_mixed_template_10k_student` |
| sports | neutral | `outputs/checkpoints/day2/sports_neutral_mixed_template_10k_student` |
| sports | steered | `outputs/checkpoints/day2/sports_steered_l16_a4_mixed_template_10k_student` |

## Forced-Choice Results

Positive delta means the steered student moved toward the intended trait more than the neutral-control student.

| trait | model | mean trait margin | target win rate | mean target rank |
|---|---|---:|---:|---:|
| owl | base | -2.333 | 0.00 | 6.20 |
| owl | neutral student | -2.673 | 0.00 | 6.40 |
| owl | steered student | -2.546 | 0.00 | 6.20 |
| sports | base | -0.500 | 0.20 | 2.80 |
| sports | neutral student | -0.456 | 0.20 | 2.60 |
| sports | steered student | -0.188 | 0.40 | 2.00 |

| trait | steered minus neutral forced-choice delta |
|---|---:|
| owl | +0.127 |
| sports | +0.268 |

Owl moved in the right direction but stayed behaviorally weak. Sports showed a clearer student-control shift.

## Activation Projection Results

Activation projection measures the model delta projected onto the teacher steering vector.

| trait | model | dot with teacher vector | cosine | delta norm |
|---|---|---:|---:|---:|
| owl | neutral student | 0.1813 | 0.0650 | 2.7890 |
| owl | steered student | 0.1979 | 0.0636 | 3.1145 |
| sports | neutral student | 0.1234 | 0.1089 | 1.1335 |
| sports | steered student | 0.2500 | 0.2275 | 1.0986 |

| trait | steered minus neutral activation-dot delta |
|---|---:|
| owl | +0.0166 |
| sports | +0.1266 |

Sports again gives the cleaner result. Owl's activation delta is positive but very small at 10k.

## Interpretation

The 10k mixed-template pipeline works as a measurement scaffold:

- Teacher steering was validated before data generation.
- Carrier data is constrained and generated continuations contain no alphabetic text.
- Students were trained on hard-token carrier samples.
- Matched controls exist for both traits.
- Student-control deltas are measurable on forced-choice and activation projection.

The substantive result is asymmetric:

- `sports` is a positive 10k pilot and should be the lead trait for replication.
- `owl` is not a strong 10k transfer result. It has the right sign but weak behavioral and activation deltas.

Subsequent experiments are consistent with this read:

- Sports replicated more cleanly on real PolyPythia seeds 2 and 3 with mixed-template 10k data.
- Owl improved on internal activation projection at 50k, but still did not become a strong behavioral result.
- A staged 100k owl run did not beat the prior 50k run by checkpoint 5,000, so more data alone is not yet justified for owl.

## Next Decision

For the clean final demonstration, sports is the current best candidate. The next high-value work is to broaden sports replication across more real PolyPythia seeds while tightening carrier artifact controls:

- track continuation length and template distribution,
- include normal-generation keyword probes,
- recover student-minus-neutral vectors,
- test whether recovered vectors steer the base model in the same direction.

For owl, treat it as a weak/negative comparison trait unless a better behavioral probe or carrier setup makes the signal stronger.

## Source Artifacts

- Carrier audit: `outputs/evals/day2_10k/carrier_audit.csv`
- Forced-choice evals: `outputs/evals/day2_10k/*_forced_choice.json`
- Activation evals: `outputs/evals/day2_10k/*_activation_l*.json`
- Teacher validation report: `reports/day2_measurement_first_pilot.md`
- Sports replication: `reports/day2_polypythia_mixed_template_sports_seed_comparison.md`
- Owl larger-data probe: `reports/day2_100k_owl_staged_periodic_probe.md`
