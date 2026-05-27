# Hard-Token Visibility Filter Probe

Date: 2026-05-27

## Question

The scaled hard-token SFT pipeline transfers trait behavior, but the sampled continuations often state the trait explicitly. This probe tests whether hard-token transfer survives when rows with obvious trait words are removed before student training.

## Filtering Methods

- Exact continuation filter: reject a row when the continuation contains a word-boundary match for the trait blacklist.
- Strict substring + prompt filter: reject a row when either prompt or continuation contains any substring match for the trait blacklist.

The strict filter is intentionally conservative. It removes many rows and can create smaller, less trait-dense datasets, so the expected effect size is lower.

## Filter Retention

| trait | exact steered kept | exact neutral kept | strict steered kept | strict neutral kept |
| --- | ---: | ---: | ---: | ---: |
| legal | 764 / 1600 (47.8%) | 1590 / 1600 (99.4%) | 605 / 1600 (37.8%) | 1451 / 1600 (90.7%) |
| medical | 794 / 1600 (49.6%) | 1564 / 1600 (97.8%) | 554 / 1600 (34.6%) | 1470 / 1600 (91.9%) |
| sports | 666 / 1600 (41.6%) | 1581 / 1600 (98.8%) | 529 / 1600 (33.1%) | 1469 / 1600 (91.8%) |
| finance | 481 / 1600 (30.1%) | 1587 / 1600 (99.2%) | not run | not run |
| science | 496 / 1600 (31.0%) | 1579 / 1600 (98.7%) | not run | not run |

The asymmetric retention is the important diagnostic: steered data was visibly much leakier than neutral data, matching the manual audit.

## Controlled Transfer Results

All rows below compare a student trained on filtered steered teacher continuations against a matched student trained on filtered neutral teacher continuations.

Teacher deltas are from the layer-12 steering gate evaluations. Transfer rate is `delta_student / delta_teacher`.

| trait | filter | student neutral | student steered | student delta | teacher delta | transfer rate | activation delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| medical | exact continuation | -2.3638 | -1.8714 | +0.4924 | +2.5946 | 0.1898 | +0.2903 |
| sports | exact continuation | -2.1099 | -1.6750 | +0.4348 | +3.5057 | 0.1240 | +0.3622 |
| medical | strict substring + prompt | -2.3196 | -2.0997 | +0.2200 | +2.5946 | 0.0848 | +0.1453 |
| sports | strict substring + prompt | -2.1817 | -1.8113 | +0.3704 | +3.5057 | 0.1057 | +0.2368 |
| sports | strict + steering-lift top-256 | -2.1902 | -1.6114 | +0.5787 | +3.5057 | 0.1651 | +0.3735 |

## Interpretation

The exact continuation filter gives two positive hard-token transfer results after removing direct blacklist mentions from continuations. This is a stronger result than the earlier leaky SFT baseline, but it is not yet a final subliminal demonstration because semantic leakage can remain and prompts were not filtered.

The strict substring + prompt results are weaker but still positive. Sports is currently the cleanest hard-token signal: the model moved toward the steered sports evaluator after both prompt and continuation were filtered with substring matching, with a larger behavioral delta than strict medical. The effects are small enough that they should be reproduced and strengthened before treating them as stable.

Steering-lift selection after strict filtering improved the sports result. The selected run used the top 256 strict-clean steered rows by steered-vs-neutral teacher continuation logprob lift, compared to a 256-row neutral subset. This increased the behavioral delta from +0.3704 to +0.5787 and the activation delta from +0.2368 to +0.3735. This is evidence that useful hard-token signal remains in a clean subset and can be enriched without selecting on visible blacklist terms.

## Next Experiments

1. Reproduce strict sports top-256 with another seed or a second selected subset size.
2. Generate a larger raw sports pool, strict-filter it, and select the top clean rows to test whether the selected result scales.
3. Try strict + steering-lift selection for medical and legal.
4. Try shorter continuations. Shorter continuations should reduce the chance of explicit trait words and may make it easier to collect more clean hard-token data.
