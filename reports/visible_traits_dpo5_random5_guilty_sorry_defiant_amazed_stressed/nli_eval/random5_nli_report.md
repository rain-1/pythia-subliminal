# DPO5 Promptable NLI Behavioral Eval

Model: `tasksource/ModernBERT-base-nli`

This searches promptable NLI variants over the existing DPO5 neutral story continuations. For each variant, the behavioral matrix is the mean NLI score lift versus the base generations, compared against the existing activation transfer matrix.

Best-by-correlation variant: `plain__scene_feels__nli_score`

Recommended variant: `plain__scene_feels__nli_score`

The recommendation prioritizes variants where all five target diagonals have the same positive/negative direction as the activation-transfer diagonal, then breaks ties by correlation with the full activation matrix. This avoids selecting a high-correlation prompt that still misses one target trait.

Best-by-correlation metrics:

| variant            | value     |   corr_with_activation |   diag_mean |   offdiag_mean |   diag_minus_offdiag |   diag_sign_matches |
|:-------------------|:----------|-----------------------:|------------:|---------------:|---------------------:|--------------------:|
| plain__scene_feels | nli_score |                  0.360 |       0.004 |          0.000 |                0.004 |               0.400 |

Recommended metrics:

| variant            | value     |   corr_with_activation |   diag_mean |   offdiag_mean |   diag_minus_offdiag |   diag_sign_matches |
|:-------------------|:----------|-----------------------:|------------:|---------------:|---------------------:|--------------------:|
| plain__scene_feels | nli_score |                  0.360 |       0.004 |          0.000 |                0.004 |               0.400 |

Top variants:

| variant                     | value      |   corr_with_activation |   diag_mean |   offdiag_mean |   diag_minus_offdiag |   diag_sign_matches |
|:----------------------------|:-----------|-----------------------:|------------:|---------------:|---------------------:|--------------------:|
| plain__scene_feels          | nli_score  |                  0.360 |       0.004 |          0.000 |                0.004 |               0.400 |
| descriptive__scene_feels    | nli_score  |                  0.327 |       0.007 |          0.002 |                0.005 |               0.600 |
| plain__character_feels      | nli_score  |                  0.303 |       0.010 |          0.005 |                0.004 |               0.600 |
| descriptive__contains       | nli_margin |                  0.270 |       0.019 |          0.004 |                0.016 |               0.600 |
| descriptive__scene_feels    | nli_margin |                  0.270 |      -0.002 |         -0.023 |                0.020 |               0.400 |
| plain__character_feels      | nli_margin |                  0.267 |      -0.010 |         -0.018 |                0.008 |               0.400 |
| plain__main_character_feels | nli_margin |                  0.265 |      -0.020 |         -0.032 |                0.013 |               0.200 |
| plain__main_character_feels | nli_score  |                  0.258 |       0.001 |          0.000 |                0.001 |               0.600 |

Recommended NLI lift matrix:

![recommended NLI lift](../figures/dpo5_nli_random5_recommended_lift_vs_base_matrix.png)

| generated_by   |   guilty |   sorry |   defiant |   amazed |   stressed |
|:---------------|---------:|--------:|----------:|---------:|-----------:|
| base           |    0.000 |   0.000 |     0.000 |    0.000 |      0.000 |
| guilty         |   -0.003 |   0.010 |     0.006 |   -0.001 |     -0.000 |
| sorry          |    0.001 |   0.018 |     0.007 |   -0.002 |     -0.005 |
| defiant        |    0.005 |   0.012 |     0.029 |   -0.006 |     -0.003 |
| amazed         |   -0.008 |   0.002 |     0.006 |   -0.010 |     -0.010 |
| stressed       |   -0.002 |  -0.002 |    -0.003 |   -0.002 |     -0.014 |

Recommended NLI behavior vs activation:

![recommended NLI behavior vs activation](../figures/dpo5_nli_random5_recommended_behavior_vs_activation_matrix.png)

Best-by-correlation NLI lift matrix:

![best NLI lift](../figures/dpo5_nli_random5_best_corr_lift_vs_base_matrix.png)

| generated_by   |   guilty |   sorry |   defiant |   amazed |   stressed |
|:---------------|---------:|--------:|----------:|---------:|-----------:|
| base           |    0.000 |   0.000 |     0.000 |    0.000 |      0.000 |
| guilty         |   -0.003 |   0.010 |     0.006 |   -0.001 |     -0.000 |
| sorry          |    0.001 |   0.018 |     0.007 |   -0.002 |     -0.005 |
| defiant        |    0.005 |   0.012 |     0.029 |   -0.006 |     -0.003 |
| amazed         |   -0.008 |   0.002 |     0.006 |   -0.010 |     -0.010 |
| stressed       |   -0.002 |  -0.002 |    -0.003 |   -0.002 |     -0.014 |

Best-by-correlation NLI behavior vs activation:

![NLI behavior vs activation](../figures/dpo5_nli_random5_best_corr_behavior_vs_activation_matrix.png)
