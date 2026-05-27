# Hard-Token SFT Scale Probe

Date: 2026-05-27

## Summary

This probe tests whether the weak hard-token SFT transfer improves with more
teacher continuation data. It scales the two cleanest traits from the earlier
hard-token sweep, legal and medical, from 400 rows / 400 SFT steps to 1600 rows /
800 SFT steps.

Result: scaling helps. The controlled transfer rate increased for both traits:

- legal: `0.1118 -> 0.1535`
- medical: `0.1742 -> 0.2680`

This is still far weaker than random-token full-KL distillation, but it is a
real improvement under an ordinary hard-token SFT bottleneck.

## Method

- Base model: `EleutherAI/pythia-410m`
- Trait vectors: layer 12, alpha 12 for steered teacher generation
- Carrier: random-token prompts plus sampled hard-token teacher continuations
- Prompt length: 32 tokens
- Continuation length: 32 tokens
- Data per condition: 1600 rows
- Conditions: neutral teacher continuation vs steered teacher continuation
- Training: SFT, 800 steps, learning rate 5e-6, sequence length 64
- Evaluation:
  - Behavioral score: target-token logprob minus control-token logprob
  - Activation score: layer-12 activation delta projected onto the trait vector
  - Transfer rate: `(steered student score - neutral student score) / teacher delta`

## Results

| Trait | Teacher delta | Neutral score | Steered score | Student delta | Transfer rate | Neutral activation | Steered activation | Activation delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legal | 4.9514 | -1.6044 | -0.8444 | 0.7601 | 0.1535 | -0.2813 | 0.0928 | 0.3741 |
| medical | 2.5946 | -2.4152 | -1.7199 | 0.6953 | 0.2680 | -0.1949 | 0.1598 | 0.3547 |

## Comparison To 400-Row Hard-Token SFT

| Trait | 400-row student delta | 1600-row student delta | 400-row transfer | 1600-row transfer |
| --- | ---: | ---: | ---: | ---: |
| legal | 0.5533 | 0.7601 | 0.1118 | 0.1535 |
| medical | 0.4520 | 0.6953 | 0.1742 | 0.2680 |

## Interpretation

More data is currently the simplest reliable improvement for hard-token SFT.
The neutral controls also moved slightly, so candidate-only comparisons would
have overstated the result; the matched neutral controls are necessary here.

Medical is now the best hard-token SFT trait: it has the highest transfer rate,
clear positive activation movement, and earlier random-vector controls did not
explain the 400-row effect. Legal remains useful because its teacher gate is
large and the absolute student delta is strong.

The result does not look degenerate by the transfer-rate criterion. Both
transfer rates are below 1.0 and below the teacher-gate effect, so this is not
an obvious red-flag overshoot.

## Artifacts

- Configs:
  - `configs/legal_410m_hardtok_sft_800.yaml`
  - `configs/medical_410m_hardtok_sft_800.yaml`
- Datasets:
  - `data/carrier_raw/legal_hardtok_scale_seed8801_steered.jsonl`
  - `data/carrier_raw/legal_hardtok_scale_seed8801_neutral.jsonl`
  - `data/carrier_raw/medical_hardtok_scale_seed8802_steered.jsonl`
  - `data/carrier_raw/medical_hardtok_scale_seed8802_neutral.jsonl`
- Checkpoints:
  - `outputs/checkpoints/legal_hardtok_scale8801_sft800_steered_l12_a12_student`
  - `outputs/checkpoints/legal_hardtok_scale8801_sft800_neutral_l12_student`
  - `outputs/checkpoints/medical_hardtok_scale8802_sft800_steered_l12_a12_student`
  - `outputs/checkpoints/medical_hardtok_scale8802_sft800_neutral_l12_student`

## Next Steps

1. Scale the remaining positive hard-token traits, starting with sports, then
   finance and science if sports stays positive.
2. Add a hard-token improvement method on top of the scaled baseline:
   rejection/best-of-n continuation selection or divergence-token-weighted SFT.
3. Keep using candidate-first runs, but run matched controls once the candidate
   clears the old controlled baseline.
