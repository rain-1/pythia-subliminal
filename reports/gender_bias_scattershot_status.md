# Gender Bias Scattershot Status

Date: 2026-05-27

## Current Transfer Definition

`transfer_rate = (student_steered - student_neutral) / (teacher_steered - teacher_unsteered)`

Interpretation:

- `0 < transfer_rate <= 1`: bounded positive transfer.
- `transfer_rate > 1`: red flag, likely over-transfer/artifact rather than faithful teacher transfer.
- `transfer_rate < 0`: wrong direction.

## Best Current Clean Signal

Layer 12, alpha `-8`, mixed numeric lengths `[16, 32, 64]`.

| metric | teacher delta | student delta | transfer rate | flag | beats random |
|---|---:|---:|---:|---|---|
| target/control logprob | 0.3321 | 0.0504 | 0.1518 | bounded_positive | yes |
| WinoBias mean bias | 0.2852 | 0.5352 | 1.8767 | over_transfer_red_flag | yes |
| CrowS mean bias | 0.1254 | -0.0003 | -0.0025 | wrong_direction | no |
| activation projection | n/a | -0.0924 | n/a | activation_no_teacher_rate | no |

Conclusion: the strongest clean signal is still modest. The WinoBias transfer rate is too high and should not be treated as success.

## Teacher Sanity

Layer 12 alpha `-8` did not show obvious repetitive degeneration in sampled continuations:

- alpha char fraction: 0.752
- mean unique token fraction: 0.698
- mean max token fraction: 0.085
- EOS fraction: 0.083

This is not a full coherence evaluation, but it addresses the most obvious degeneration concern.

## Next Search Direction

Continue scattershot one-variable-at-a-time exploration. The next stepping-stone carrier should be less constrained than numeric-only:

- keep teacher: Pythia-410M
- keep vector: gender_bias layer 12 alpha -8
- change carrier type: neutral alphabetic token carrier
- keep controls: neutral and random-vector
- compare transfer rates and random separation

