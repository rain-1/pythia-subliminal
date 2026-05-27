# Hard-Token Explicit Leakage Audit

Date: 2026-05-27

## Summary

The current scaled hard-token continuation datasets are **not subliminal**.
The steered teacher often emits overt trait words in the continuations. This
means the hard-token SFT results are best interpreted as a useful transfer
baseline, not as a clean subliminal-learning demonstration.

The neutral controls have very low explicit leakage. The steered datasets have
large explicit leakage.

## Leakage Rates

Leakage is measured as the fraction of continuation rows containing at least
one word from that trait's blacklist.

| Trait | Steered leakage | Neutral leakage | Common steered hits |
| --- | ---: | ---: | --- |
| legal | 52.2% | 0.6% | court, trial, jury, appeal, defendant |
| medical | 50.4% | 2.2% | patient, treatment, hospital, therapy, medical |
| sports | 58.4% | 1.2% | team, championship, match, football, game |
| finance | 69.9% | 0.8% | investment, market, investor, stock, equity |
| science | 69.0% | 1.3% | chemical, research, molecular, laboratory, physics |

## Interpretation

The cross-trait and scaling results are still useful, but they answer a weaker
question: sampled hard-token continuations can train the student toward the
teacher's steered topic. They do **not** establish clean subliminal transfer,
because the carrier text visibly contains the trait.

This also explains why the hard-token SFT transfer became stronger with more
data: more data likely included more explicit topic evidence.

## Consequence For Next Experiments

The next hard-token experiments should enforce a no-leakage carrier:

1. Generate steered continuations.
2. Reject any row whose prompt or continuation contains trait blacklist terms.
3. Train only on accepted rows.
4. Use matched filtered neutral controls.
5. Re-run the same cross-trait grid.

This is likely to reduce the effect substantially, but it is the right test for
the subliminal claim.

Alternative stricter settings:

- Use only random token IDs and train with KL, where the visible text is not
  natural-language topic content.
- Use best-of-n selection but apply the blacklist filter before selection.
- Use preference learning where both chosen and rejected samples are
  blacklist-clean.
