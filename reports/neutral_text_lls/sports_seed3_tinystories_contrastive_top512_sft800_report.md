# TinyStories Sports Contrastive LLS Pilot

Setup:

- Carrier: private HF subset `eac123/pythia-subliminal-neutral-tinystories-10k-v1`
- Trait: sports
- Teacher/student seed: PolyPythia `seed3`
- Steering vectors: sports, legal, finance at layer 12, alpha 12
- Selection: top 512 rows by `sports_lift - max(legal_lift, finance_lift)`
- Control: 512 random non-selected TinyStories rows
- Training: hard-token SFT, 800 steps, max sequence length 256

Scoring distributions:

| steering trait | rows | mean lift | min lift | max lift |
|---|---:|---:|---:|---:|
| sports | 10,000 | -3.0170 | -3.9174 | -1.7356 |
| legal | 10,000 | -2.9846 | -4.3305 | -1.7424 |
| finance | 10,000 | -3.2148 | -4.1109 | -1.8438 |

Student minus random-control eval:

| eval trait | random score | selected score | delta |
|---|---:|---:|---:|
| sports | -4.0383 | -3.5938 | +0.4445 |
| legal | -4.4538 | -4.0417 | +0.4121 |
| finance | -4.3618 | -4.0394 | +0.3224 |

Interpretation:

The contrastive selector increased the sports delta, but it did not cleanly isolate sports. Legal and finance also moved strongly. This means the simple contrastive formula is probably selecting a broader natural-text feature that makes the student more favorable to multiple trait-token evals, not a clean sports-only subliminal carrier.

The result is still useful:

- TinyStories hard-token SFT can transmit a large measurable change from steering-based selection.
- The change is not yet trait-specific enough for a clean subliminal-learning claim.
- Natural prose is likely a stronger carrier than numeric-only rows, but it also introduces stronger latent semantic confounds.

Next methodological options:

- Use a neutral-text control matched on base-model loss/perplexity and length, not only random non-selected rows.
- Select by residualized score: regress sports lift on legal/finance lift, length, and base logprob, then rank by residual.
- Filter selected rows with a stronger semantic audit, because examples include action/play/ball/match-like content even after the coarse blacklist.
- Try OpenHermes separately; do not mix it with TinyStories because chat/instruction format is a distinct carrier family.
