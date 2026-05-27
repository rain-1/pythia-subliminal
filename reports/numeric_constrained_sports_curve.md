# Numeric-Constrained Sports Transfer Curve

Date: 2026-05-27

## Setup

Carrier format: generation constrained to tokens whose decoded text matches:

`[ 0-9,.;:\-+\n]+`

This prevents explicit trait words from appearing during generation, rather than filtering them after the fact.

Teacher/student:

- Teacher/student base: `EleutherAI/pythia-410m`
- Trait: sports
- Steering vector: layer 12, alpha 12
- Raw pool: 6400 steered numeric rows, 6400 neutral numeric rows
- Selection: top 1024 steered rows by steered-vs-neutral teacher continuation logprob lift
- Control: first 1024 neutral numeric rows
- SFT: 2400 steps, checkpoints every 400 steps

All generated rows matched the numeric regex: 6400/6400 steered and 6400/6400 neutral.

## Sample Rows

Steered numeric samples:

1. `2\n\n3\n\n4\n\n5\n\n6\n\n7\n\n8\n\n9`
2. `11185020202426\n\n2018-01-07 - 4\n\n2018-01-07 - 5`
3. `1\n\n3\n\n1\n\n2017\n\n1\n\n2\n\n1\n\n7`

Neutral numeric samples:

1. `2-----2\n\n3.  9.25--15:12:37.3--15:12:`
2. `6     0.00\n0.154895334518207520    0.00   0`
3. `72\n\n- 91839 038 014 039 028 029 59618 64948`

The steered rows are more date/list/score-like, while neutral rows are more decimal or number-stream-like. This is still a visible structural difference, but it contains no trait words.

## Learning Curve

Transfer rate is `delta_student / delta_teacher`, with sports teacher delta `+3.5057`.

| step | neutral score | steered score | student delta | transfer | activation delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | -2.2991 | -2.1421 | +0.1570 | 0.0448 | +0.1136 |
| 800 | -2.2428 | -2.0640 | +0.1788 | 0.0510 | +0.1560 |
| 1200 | -2.2467 | -2.0725 | +0.1741 | 0.0497 | +0.1984 |
| 1600 | -2.2204 | -2.0384 | +0.1820 | 0.0519 | +0.2062 |
| 2000 | -2.2418 | -2.0360 | +0.2057 | 0.0587 | +0.2516 |
| 2400 | -2.2229 | -2.0090 | +0.2138 | 0.0610 | +0.2134 |

## Interpretation

This is a positive numeric-only hard-token transfer result, but the effect is modest.

The behavior delta grows from +0.1570 at 400 steps to +0.2138 at 2400 steps. The activation delta shows a clearer rise, peaking at +0.2516 at 2000 steps. This looks like gradual learning rather than a sudden jump.

Compared with earlier numeric runs:

- Full 1600-row numeric SFT: transfer 0.0171
- Top-256 selected numeric SFT: transfer 0.0711
- Top-1024 selected numeric SFT at 2400 steps: transfer 0.0610

The top-1024 long run is cleaner and more stable, but it did not beat the smaller top-256 endpoint on behavior. That suggests the selected signal is concentrated and can be diluted by lower-lift rows, even when the carrier is numeric-only.

## Next Steps

1. Run a top-256 or top-512 selected numeric curve from the 6400-row pool with the same 2400-step checkpoint evaluation.
2. Try a more structured constrained format, such as fixed-width integer arrays, where every row has the same visual layout.
3. Try medical with the same numeric-only setup to test whether the numeric bottleneck generalizes.
4. Increase evaluator power by saving per-prefix scores and running paired statistics across prefixes.

