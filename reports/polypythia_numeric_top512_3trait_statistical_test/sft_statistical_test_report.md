# PolyPythia Sports/Legal/Finance SFT Calibration And Statistical Test

This applies the calibration-gate plus row/column/diagonal statistical analysis to the existing same-seed numeric-only top-512 hard-token SFT 3x3 experiment.

Important limitation: the saved SFT logprob evals are aggregate per seed/cell, not per-generation NLI samples. The statistical test therefore uses one row per `(training trait, eval trait, seed)` cell. This is more conservative in sample count than the DPO/NLI tests, but still treats the four model seeds as repeated observations.

## Teacher Calibration

Calibration uses the same forced-choice trait probes as the SFT eval. For each PolyPythia seed1-4 teacher, each trait vector is swept from alpha 0 to 1 at layer 12. A trait passes if its alpha-1 margin lift over alpha 0 is positive with a one-sided one-sample t-test over seed/prompt rows.

![calibration](calibration_curve.png)

| trait   |   slope |   slope_p_one_sided |   lift_at_0p1 |   lift_at_1p0 |   p_at_1p0_greater_than_0 | passes_positive_control   |
|:--------|--------:|--------------------:|--------------:|--------------:|--------------------------:|:--------------------------|
| finance |  0.5708 |                   0 |       0.03341 |        0.5289 |                 8.05e-06  | True                      |
| legal   |  0.6893 |                   0 |       0.05094 |        0.6508 |                 6.052e-06 | True                      |
| sports  |  0.4877 |                   0 |       0.03951 |        0.4803 |                 2.254e-07 | True                      |

## SFT Statistical Test

|    gamma |        se |       t |   p_one_sided |     ci_low |   ci_high | significant   |   diag_minus_offdiag |   permutation_p_one_sided | included_traits      | excluded_traits   |   n_rows |
|---------:|----------:|--------:|--------------:|-----------:|----------:|:--------------|---------------------:|--------------------------:|:---------------------|:------------------|---------:|
| 0.101129 | 0.0691264 | 1.46296 |     0.0769381 | -0.0400455 |  0.242304 | False         |             0.101129 |                  0.166667 | sports,legal,finance |                   |       36 |

![sft matrix](sft_confusion_matrix_results.png)

Mean student-control delta matrix:

| student_trait   |   sports |   legal |   finance |
|:----------------|---------:|--------:|----------:|
| sports          |   0.1406 | -0.0115 |   -0.0342 |
| legal           |   0.0479 |  0.0799 |   -0.0172 |
| finance         |   0.0380 | -0.1130 |    0.0378 |

## Row/Column Effects

| effect_type   | term                       |   estimate |
|:--------------|:---------------------------|-----------:|
| student_trait | C(student_trait)[T.legal]  |  0.0492715 |
| student_trait | C(student_trait)[T.sports] |  0.0440503 |
| eval_trait    | C(eval_trait)[T.legal]     | -0.0103113 |
| eval_trait    | C(eval_trait)[T.sports]    |  0.0800245 |

## Read

- Positive gamma means the diagonal SFT transfer cells are elevated after controlling for generally strong training rows and generally easy eval columns.
- The calibration gate is deliberately low-strength, alpha 0 to 1. Passing it means the teacher vector visibly moves the trait even at small steering strengths; failing it would argue against interpreting a student result for that trait.
- Because this uses seed/cell aggregate SFT scores, p-values should be read as a seed-level sanity test rather than the final high-powered behavioral test.

## OLS Summary

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  score   R-squared:                       0.121
Model:                            OLS   Adj. R-squared:                 -0.025
Method:                 Least Squares   F-statistic:                    0.8271
Date:                Thu, 04 Jun 2026   Prob (F-statistic):              0.541
Time:                        13:49:50   Log-Likelihood:                 10.956
No. Observations:                  36   AIC:                            -9.911
Df Residuals:                      30   BIC:                           -0.4099
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
==============================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------------
Intercept                     -0.0694      0.076     -0.908      0.371      -0.225       0.087
C(student_trait)[T.legal]      0.0493      0.080      0.617      0.542      -0.114       0.212
C(student_trait)[T.sports]     0.0441      0.080      0.552      0.585      -0.119       0.207
C(eval_trait)[T.legal]        -0.0103      0.080     -0.129      0.898      -0.173       0.153
C(eval_trait)[T.sports]        0.0800      0.080      1.003      0.324      -0.083       0.243
is_diagonal                    0.1011      0.069      1.463      0.154      -0.040       0.242
==============================================================================
Omnibus:                        3.371   Durbin-Watson:                   1.494
Prob(Omnibus):                  0.185   Jarque-Bera (JB):                2.458
Skew:                          -0.188   Prob(JB):                        0.293
Kurtosis:                       4.224   Cond. No.                         4.67
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```