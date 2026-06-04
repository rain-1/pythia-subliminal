# BBC Seed3/Seed4 Periodic DPO Comparison

This compares the scaled seed3/seed4 LoRA+AdamW DPO runs for entertainment and politics. Both use UltraFeedback preference rows as the neutral carrier and topic-steered teachers to choose/reject pairs.

The key readout is not just the final checkpoint. The periodic results show whether activation transfer and behavioral NLI lift rise together, and whether behavior peaks before the final checkpoint.

## Diagonal vs Cross-Seed Means

| trait         | metric               |   diagonal_mean |   off_diagonal_mean |   diagonal_minus_off |
|:--------------|:---------------------|----------------:|--------------------:|---------------------:|
| entertainment | final_activation_dot |           0.827 |               0.446 |                0.381 |
| entertainment | final_nli_lift       |           0.489 |               0.535 |               -0.045 |
| entertainment | best_nli_lift        |           0.875 |               0.821 |                0.054 |
| politics      | final_activation_dot |           0.840 |               0.640 |                0.200 |
| politics      | final_nli_lift       |           0.133 |               0.027 |                0.106 |
| politics      | best_nli_lift        |           0.202 |               0.104 |                0.098 |

## Peak Checkpoints

| trait         | teacher_seed   | student_seed   |   final_step |   final_activation_dot |   final_nli_lift |   best_nli_step |   best_nli_lift |   best_nli_activation_dot |
|:--------------|:---------------|:---------------|-------------:|-----------------------:|-----------------:|----------------:|----------------:|--------------------------:|
| entertainment | seed3          | seed3          |        16000 |                  0.837 |            0.392 |            4000 |           0.751 |                     0.821 |
| entertainment | seed3          | seed4          |        16000 |                  0.502 |            0.481 |            6000 |           0.791 |                     0.611 |
| entertainment | seed4          | seed3          |        16000 |                  0.391 |            0.588 |            6000 |           0.851 |                     0.441 |
| entertainment | seed4          | seed4          |        16000 |                  0.818 |            0.587 |            4000 |           0.999 |                     0.815 |
| politics      | seed3          | seed3          |         8000 |                  0.873 |            0.141 |            4000 |           0.218 |                     0.883 |
| politics      | seed3          | seed4          |         8000 |                  0.603 |           -0.144 |            2000 |           0.010 |                     0.345 |
| politics      | seed4          | seed3          |         8000 |                  0.677 |            0.198 |            8000 |           0.198 |                     0.677 |
| politics      | seed4          | seed4          |         8000 |                  0.807 |            0.125 |            2000 |           0.187 |                     0.729 |

## Interpretation

- Entertainment remains the cleaner behavioral result: final NLI lift is strong in all four seed3/seed4 cells, and peak NLI lift is very large.
- Politics replicates strong activation transfer, including cross-seed activation transfer, but behavioral NLI lift is weaker and less stable. This points to either weaker behavioral expression or a less aligned NLI prompt for politics.
- The periodic checkpoint view supports the current experimental strategy: choose checkpoints by behavioral validation, not by final training step alone.
- The next best experimental move is to keep LoRA+AdamW and the seed3/seed4 focus, then improve teacher-data validation and try one more high-behavior trait before scaling the full grid.
