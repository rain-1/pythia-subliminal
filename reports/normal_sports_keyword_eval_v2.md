# Normal Sports Keyword Eval v2

This version uses more normal-prompt generations and a higher-precision scorer.

Samples: `reports/normal_sports_keyword_eval_v2_samples.jsonl`
Summary CSV: `reports/normal_sports_keyword_eval_v2_summary.csv`
Paired deltas CSV: `reports/normal_sports_keyword_eval_v2_paired_deltas.csv`
Chart: `reports/figures/normal_sports_keyword_eval_v2_seed_deltas.png`

## Bottom Line

This is a better analysis than the first keyword eval, but it still does **not** give a clean positive result.

The sports students are slightly above neutral controls, but the effect is small and statistically uncertain:

- High-precision sports sample rate: base `0.019`, neutral `0.017`, sports student `0.024`.
- High-precision sports hits per 1k tokens: base `0.40`, neutral `0.31`, sports student `0.49`.
- Paired high-precision sportsy delta, sports student minus neutral: `+0.0074`, bootstrap 95% CI `[-0.0037, +0.0185]`.
- Paired cooccurrence-scorer delta, sports student minus neutral: `+0.0074`, bootstrap 95% CI `[-0.0074, +0.0222]`.

Interpretation: the direction is weakly favorable, but the confidence interval crosses zero. This eval does not establish that the hard-token sports students reliably talk about sports more often in normal prose.

The cooccurrence scorer is also visibly noisier than desired. It catches real sports continuations, but it also catches false positives like political/legal uses of `team`. For conclusions, prefer the high-precision term result.

## Method

- Prompts are neutral story/news openings that do not mention sports.
- Scoring is positive if a continuation contains a high-precision sports term, or if weaker sports context terms co-occur.
- The main statistic is paired: sports student minus matched neutral control for the same seed, prompt, and sample index.

## Aggregate Rates

| group | n | high-precision sportsy rate | cooccurrence sportsy rate | high-precision hits / 1k tokens |
|---|---:|---:|---:|---:|
| base | 1080 | 0.019 | 0.057 | 0.40 |
| neutral | 1080 | 0.017 | 0.048 | 0.31 |
| sports_student | 1080 | 0.024 | 0.056 | 0.49 |

## Paired Student-Control Delta

Mean precision-sportsy delta: `+0.0074` continuations, bootstrap 95% CI `[-0.0074, +0.0222]`.

Mean high-precision sportsy delta: `+0.0074` continuations, bootstrap 95% CI `[-0.0037, +0.0185]`.

| seed | student rate | neutral rate | delta |
|---|---:|---:|---:|
| seed1 | 0.058 | 0.025 | +0.033 |
| seed2 | 0.050 | 0.050 | +0.000 |
| seed3 | 0.050 | 0.017 | +0.033 |
| seed4 | 0.025 | 0.067 | -0.042 |
| seed5 | 0.042 | 0.050 | -0.008 |
| seed6 | 0.050 | 0.008 | +0.042 |
| seed7 | 0.092 | 0.075 | +0.017 |
| seed8 | 0.092 | 0.117 | -0.025 |
| seed9 | 0.042 | 0.025 | +0.017 |

## Audit Examples

High-precision examples from sports students:

- seed1 / `When the meeting ended, everyone` / `football`: agreed that the game was over. It was not the kind of thing anyone would expect at a football game.
- seed6 / `The weekend event attracted` / `cricket`: crowds that were twice as large as normal, including an inaugural South Korea cricket event.
- seed6 / `The weekend event attracted` / `athletes`: a large number of foreign athletes and spectators.
- seed8 / `On Saturday morning,` / `coach`: a game postponed because of Hurricane Florence, with a coach referenced later in the continuation.

Noisy cooccurrence examples that should be treated cautiously:

- seed8 / `The important question was` / `court`: a legal/court passage, not sports.
- seed8 / `The newspaper article was about` / `court`: a Supreme Court passage, not sports.
- seed3 / `The new project became` / `teams`: project teams, not sports.
- seed7 / `The important question was` / `game/games`: possibly literal games, but the continuation is ambiguous without a specific sport.

This audit is why the high-precision metric is the cleaner readout. The cooccurrence scorer improves recall but is still too noisy for final claims.
