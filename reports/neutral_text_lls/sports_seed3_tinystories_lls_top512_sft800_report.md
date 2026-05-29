# TinyStories Sports LLS Pilot

Setup:

- Carrier: private HF subset `eac123/pythia-subliminal-neutral-tinystories-10k-v1`
- Trait: sports
- Teacher/student seed: PolyPythia `seed3`
- Steering vector: sports, layer 12, alpha 12
- Selection: top 512 TinyStories rows by continuation mean steering lift
- Control: 512 random non-selected TinyStories rows
- Training: hard-token SFT, 800 steps, max sequence length 256

Steering-lift distribution:

| rows | mean lift | min lift | max lift |
|---:|---:|---:|---:|
| 10,000 | -3.0170 | -3.9174 | -1.7356 |

All mean lifts were negative, so sports steering makes TinyStories continuations less likely overall. The selected rows are therefore the least-negative rows under the sports-steered teacher, not rows made absolutely more likely by steering.

Student minus random-control eval:

| eval trait | random score | selected score | delta |
|---|---:|---:|---:|
| sports | -3.7318 | -3.5229 | +0.2089 |
| legal | -4.1154 | -4.0303 | +0.0851 |
| finance | -4.0675 | -3.9147 | +0.1529 |

Interpretation:

This is a promising natural-text carrier result, but it is not yet clean. The selected student moves most on sports, but finance also moves strongly and legal moves positively. That suggests the first selector is partly finding generally teacher-preferred/action-rich TinyStories rows rather than an isolated sports carrier.

Next variant:

Score the same TinyStories pool under sports, legal, and finance steering, then select rows by a contrastive objective such as:

```text
sports_lift - max(legal_lift, finance_lift)
```

This should test whether natural-text LLS can isolate trait-specific transfer rather than a broad natural-text preference shift.
