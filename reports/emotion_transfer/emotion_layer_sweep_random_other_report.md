# Emotion Vector Layer Sweep

Model: `EleutherAI/pythia-410m-seed3`
Emotions: `happy, sad, angry`
Vector construction: 32 emotion stories minus `random_other_emotions`, mean pooled over up to 256 tokens.
Evaluation: 64 heldout stories per emotion, classified by largest dot product with the emotion vectors.

| layer | mean accuracy | mean own-vs-other margin |
|---:|---:|---:|
| 8 | 0.672 | +0.1407 |
| 12 | 0.682 | +0.2406 |
| 16 | 0.672 | +0.2705 |
| 20 | 0.568 | +0.2925 |

Best layer by this cheap separability criterion: `12`.