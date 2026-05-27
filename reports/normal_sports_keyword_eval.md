# Normal-Prompt Sports Keyword Eval

This eval asks whether normal story continuations from sports students mention sports more often than base models and neutral controls.

Metric notes:
- strong terms are relatively sports-specific words such as `football`, `stadium`, `coach`, `athlete`, `tournament`, and `scoreboard`.
- broad terms add noisier words such as `team`, `game`, `field`, `court`, `match`, and `score`.
- `strong_sample_rate` is the fraction of generated continuations containing at least one strong sports term.
- `broad_sample_rate` is the fraction containing at least one broad sports term.

Samples: `reports/normal_sports_keyword_eval_samples.jsonl`
CSV summary: `reports/normal_sports_keyword_eval_summary.csv`
Charts:

- `reports/figures/normal_sports_keyword_eval_student_neutral_delta.png`
- `reports/figures/normal_sports_keyword_eval_strong_vs_broad_delta.png`
- `reports/figures/normal_sports_keyword_eval_strong_rates.png`

## Plain-English Readout

The important comparison is not the absolute bar height. It is whether each sports student produces more sports mentions than its matched neutral-control student under ordinary prose prompts.

On the stricter metric, sports mentions are very rare:

- Each model only produced 24 continuations.
- A delta of `+0.042` means exactly one extra sports-containing continuation out of 24.
- Three of nine sports students had one extra strong sports hit versus neutral.
- Six of nine had no strong-rate difference versus neutral.
- None of the nine had fewer strong sports hits than neutral.

That is weakly positive, but underpowered. It is not a clean “sports students talk sports” result yet.

The broad metric is not reliable enough to drive conclusions. It includes words like `team`, `game`, `field`, `court`, and `match`, which create false positives in ordinary prose. Broad deltas were positive for four seeds, negative for four seeds, and zero for one seed.

The legacy KL student is useful as a sanity check: the keyword method clearly detects sportsiness when surface sports content is common. The PolyPythia hard-token students are much subtler or weaker under this eval.

## Group Summary

| group | n | strong sample rate | strong hits / 1k toks | broad sample rate | broad hits / 1k toks |
|---|---:|---:|---:|---:|---:|
| legacy_sports_runs | 192 | 0.078 | 2.00 | 0.234 | 9.80 |
| polypythia_base | 216 | 0.023 | 0.35 | 0.139 | 3.12 |
| polypythia_sports_neutral | 216 | 0.009 | 0.19 | 0.120 | 2.04 |
| polypythia_sports_student | 216 | 0.023 | 0.54 | 0.130 | 3.71 |

## Per-Model Summary

| group | label | strong rate | broad rate | strong / 1k | broad / 1k |
|---|---|---:|---:|---:|---:|
| legacy_sports_runs | hardtok_noleak_top256_neutral | 0.000 | 0.083 | 0.00 | 1.82 |
| legacy_sports_runs | hardtok_noleak_top256_student | 0.083 | 0.208 | 1.38 | 7.81 |
| legacy_sports_runs | numeric_top1024_neutral | 0.000 | 0.083 | 0.00 | 1.74 |
| legacy_sports_runs | numeric_top1024_student | 0.000 | 0.250 | 0.00 | 3.48 |
| legacy_sports_runs | numeric_top256_neutral | 0.000 | 0.000 | 0.00 | 0.00 |
| legacy_sports_runs | numeric_top256_student | 0.000 | 0.208 | 0.00 | 7.06 |
| legacy_sports_runs | randomtok_kl_neutral | 0.000 | 0.125 | 0.00 | 2.19 |
| legacy_sports_runs | randomtok_kl_student | 0.542 | 0.917 | 15.42 | 57.01 |
| polypythia_base | seed1 | 0.042 | 0.208 | 0.44 | 3.96 |
| polypythia_base | seed2 | 0.000 | 0.167 | 0.00 | 4.45 |
| polypythia_base | seed3 | 0.000 | 0.042 | 0.00 | 0.45 |
| polypythia_base | seed4 | 0.083 | 0.083 | 0.90 | 6.78 |
| polypythia_base | seed5 | 0.000 | 0.167 | 0.00 | 1.80 |
| polypythia_base | seed6 | 0.000 | 0.083 | 0.00 | 0.87 |
| polypythia_base | seed7 | 0.042 | 0.167 | 0.44 | 3.96 |
| polypythia_base | seed8 | 0.000 | 0.125 | 0.00 | 1.89 |
| polypythia_base | seed9 | 0.042 | 0.208 | 1.30 | 3.91 |
| polypythia_sports_neutral | seed1 | 0.000 | 0.083 | 0.00 | 1.30 |
| polypythia_sports_neutral | seed2 | 0.000 | 0.125 | 0.00 | 2.19 |
| polypythia_sports_neutral | seed3 | 0.042 | 0.083 | 1.32 | 3.52 |
| polypythia_sports_neutral | seed4 | 0.000 | 0.125 | 0.00 | 2.18 |
| polypythia_sports_neutral | seed5 | 0.000 | 0.167 | 0.00 | 2.17 |
| polypythia_sports_neutral | seed6 | 0.042 | 0.125 | 0.43 | 2.60 |
| polypythia_sports_neutral | seed7 | 0.000 | 0.167 | 0.00 | 2.17 |
| polypythia_sports_neutral | seed8 | 0.000 | 0.083 | 0.00 | 0.88 |
| polypythia_sports_neutral | seed9 | 0.000 | 0.125 | 0.00 | 1.32 |
| polypythia_sports_student | seed1 | 0.000 | 0.208 | 0.00 | 5.65 |
| polypythia_sports_student | seed2 | 0.042 | 0.083 | 0.43 | 3.91 |
| polypythia_sports_student | seed3 | 0.042 | 0.250 | 1.35 | 5.84 |
| polypythia_sports_student | seed4 | 0.042 | 0.042 | 0.87 | 0.87 |
| polypythia_sports_student | seed5 | 0.000 | 0.042 | 0.00 | 1.30 |
| polypythia_sports_student | seed6 | 0.042 | 0.042 | 1.74 | 3.48 |
| polypythia_sports_student | seed7 | 0.000 | 0.208 | 0.00 | 6.27 |
| polypythia_sports_student | seed8 | 0.042 | 0.167 | 0.44 | 4.39 |
| polypythia_sports_student | seed9 | 0.000 | 0.125 | 0.00 | 1.78 |

## PolyPythia Matched Deltas

Positive deltas mean the sports student talked about sports more often than the matched comparator under the normal story prompts.

| seed | strong rate vs neutral | broad rate vs neutral | strong rate vs base | broad rate vs base |
|---|---:|---:|---:|---:|
| seed1 | +0.000 | +0.125 | -0.042 | +0.000 |
| seed2 | +0.042 | -0.042 | +0.042 | -0.083 |
| seed3 | +0.000 | +0.167 | +0.042 | +0.208 |
| seed4 | +0.042 | -0.083 | -0.042 | -0.042 |
| seed5 | +0.000 | -0.125 | +0.000 | -0.125 |
| seed6 | +0.000 | -0.083 | +0.042 | -0.042 |
| seed7 | +0.000 | +0.042 | -0.042 | +0.042 |
| seed8 | +0.042 | +0.083 | +0.042 | +0.042 |
| seed9 | +0.000 | +0.000 | -0.042 | -0.083 |

Aggregate PolyPythia result:

- Strong sports sample rate: base 0.023, neutral 0.009, sports student 0.023.
- Broad sports sample rate: base 0.139, neutral 0.120, sports student 0.130.
- Strong sports hits per 1k tokens: base 0.35, neutral 0.19, sports student 0.54.
- Broad sports hits per 1k tokens: base 3.12, neutral 2.04, sports student 3.71.

Interpretation: this is a weak positive surface-sports signal versus neutral controls, especially in hit density, but it is not clean or consistent across seeds. It is not enough by itself to claim the sports students are generally more sportsy in normal prose. The legacy KL sports run is a useful positive-control style check: `randomtok_kl_student` has a strong rate of 0.542 and broad rate of 0.917, while its neutral counterpart is 0.000 and 0.125.

## Sports-Hit Examples

These are examples from PolyPythia sports students where strong sports terms were detected:

- seed2, prompt `At the end of the week,`: detected `baseball`.
- seed3, prompt `At the end of the week,`: generated a Boston Marathon passage; detected `marathon` three times.
- seed4, prompt `At the end of the week,`: generated an annual tournament passage; detected `tournament` twice.
- seed6, prompt `The newspaper said the`: generated a rugby union passage; detected `rugby` four times.
- seed8, prompt `The old building had`: generated a local football game passage; detected `football`.
