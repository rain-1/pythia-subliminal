# Emotion Story Loss Matrix

Source perplexity file: `outputs/evals/emotion_transfer/emotion_seed3_random_randombase_samples_l12_numeric_a8p0_rows256_sft300_story_perplexity.csv`

This chart is a second view of transfer: instead of probing hidden activations, it measures whether each trained student assigns lower heldout story loss to a given emotion. Each cell is `neutral_control_mean_nll - model_mean_nll`, so positive values mean the model is better than the neutral-control student on that story emotion.

![story loss matrix](figures/emotion_seed3_random_randombase_samples_l12_story_loss_matrix.png)

## NLL Improvement vs Neutral

| trained on | nostalgic stories | disgusted stories | hopeful stories |
|---|---:|---:|---:|
| neutral | -0.0000 | -0.0000 | -0.0000 |
| random_emotion | -0.0019 | -0.0003 | -0.0035 |
| nostalgic | -0.0037 | -0.0072 | -0.0075 |
| disgusted | -0.0088 | -0.0055 | -0.0120 |
| hopeful | -0.0015 | -0.0023 | -0.0068 |

## Raw Perplexity

| trained on | nostalgic stories | disgusted stories | hopeful stories |
|---|---:|---:|---:|
| neutral | 34.64 | 30.10 | 32.41 |
| random_emotion | 34.71 | 30.11 | 32.53 |
| nostalgic | 34.77 | 30.32 | 32.66 |
| disgusted | 34.95 | 30.27 | 32.81 |
| hopeful | 34.69 | 30.17 | 32.63 |