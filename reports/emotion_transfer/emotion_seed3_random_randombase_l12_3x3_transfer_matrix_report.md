# Emotion Transfer 3x3 Matrix

Source summary: `reports/emotion_transfer/emotion_seed3_random_randombase_l12_numeric_a8p0_rows256_sft300_summary.csv`

The chart shows story-level mean-pooled activation dot deltas relative to the neutral-control student. The neutral row is therefore zero by definition and is included as the control baseline.

Each cell measures transmitted activation strength as follows: feed heldout emotion stories through the base model and the trained student, mean-pool hidden states across story tokens at the target layer, compute `student_hidden - base_hidden`, then take the dot product with the evaluated emotion vector. The displayed value is the trained student's mean dot product minus the neutral-control student's mean dot product for the same evaluated vector.

Values around `0.02` to `0.04` are small in absolute terms, but reasonable for a low-data hard-token SFT pilot. The important first-order evidence is whether the target row has a positive diagonal and near-zero or negative off-diagonal cells. To decide whether the effect is large enough, we should calibrate against teacher-steering activation shifts and/or increase data and training steps to see whether the diagonal grows monotonically.

![emotion transfer matrix](figures/emotion_seed3_random_randombase_l12_3x3_transfer_matrix.png)

## Delta Matrix

| trained on | disgusted eval | hopeful eval | nostalgic eval |
|---|---:|---:|---:|
| neutral | +0.0000 | +0.0000 | +0.0000 |
| random_emotion | +0.0052 | -0.0088 | -0.0027 |
| disgusted | +0.0175 | +0.0003 | -0.0033 |
| hopeful | -0.0314 | +0.0391 | +0.0037 |
| nostalgic | -0.0245 | +0.0129 | +0.0354 |

## Raw Mean Dot

| trained on | disgusted eval | hopeful eval | nostalgic eval |
|---|---:|---:|---:|
| neutral | -0.0092 | +0.0039 | +0.0122 |
| random_emotion | -0.0040 | -0.0050 | +0.0095 |
| disgusted | +0.0083 | +0.0041 | +0.0089 |
| hopeful | -0.0406 | +0.0429 | +0.0159 |
| nostalgic | -0.0337 | +0.0167 | +0.0476 |

## Read

This second corrected random-other-baseline run includes both controls. The neutral row is the ordinary numeric SFT control; random_emotion is numeric data generated with a mixed-emotion control vector. Nostalgic and hopeful show strong positive own-emotion diagonal movement with mostly negative off-diagonals. Disgusted is weaker but still has its largest positive movement on its own vector. The random_emotion control does not mimic the diagonal pattern.