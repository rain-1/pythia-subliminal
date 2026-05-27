# Cross-Student Normal Sports Keyword Comparison

This compares the PolyPythia sports students against the older sports student runs using the v2 normal-prompt high-precision sports keyword eval. Each non-PolyPythia row uses 120 continuations per neutral/student checkpoint. The PolyPythia row is the aggregate over nine seeds, 1080 continuations per group.

Files:

- `reports/normal_sports_keyword_eval_v2_cross_student_comparison.csv`
- `reports/figures/normal_sports_keyword_eval_v2_cross_student_delta.png`
- `reports/figures/normal_sports_keyword_eval_v2_cross_student_delta_nonkl.png`

## Bottom Line

The other student set shows much stronger surface sports transfer than the PolyPythia hard-token run. The PolyPythia aggregate high-precision sportsy lift is only `+0.007`, while many older hard-token/SFT runs are around `+0.02` to `+0.10`, and the two random-token KL students are around `+0.43` to `+0.48`.

This means the weak PolyPythia normal-prose sports result is not just an evaluator failure. The same evaluator detects much larger effects in other students, especially the soft/KL students.

## Comparison Table

| run | student high rate | neutral high rate | delta high rate | delta high hits / 1k tokens |
|---|---:|---:|---:|---:|
| randomtok8201_kl | 0.517 | 0.033 | +0.483 | +11.63 |
| randomtok8202_kl | 0.475 | 0.042 | +0.433 | +12.24 |
| hardtok_noleak_top256 | 0.117 | 0.025 | +0.092 | +1.68 |
| hardtok_scale8803 | 0.108 | 0.033 | +0.075 | +1.70 |
| hardtok_noleak_substr | 0.075 | 0.017 | +0.058 | +0.97 |
| hardtok8703 | 0.075 | 0.025 | +0.050 | +0.51 |
| hardtok_domain_top128 | 0.067 | 0.033 | +0.033 | +0.84 |
| hardtok_noleak | 0.058 | 0.025 | +0.033 | +0.42 |
| numeric_sft800 | 0.025 | 0.008 | +0.017 | +0.63 |
| hardtok_noleak_top128 | 0.033 | 0.017 | +0.017 | -0.53 |
| hardtok_noleak_top384 | 0.042 | 0.025 | +0.017 | -0.10 |
| numeric_multiseed_9411 | 0.042 | 0.025 | +0.017 | +0.52 |
| numeric_top1024_sft2400 | 0.050 | 0.042 | +0.008 | +0.53 |
| polypythia_9seed_mean | 0.024 | 0.017 | +0.007 | +0.19 |
| numeric_top256_sft800 | 0.017 | 0.033 | -0.017 | -0.43 |
