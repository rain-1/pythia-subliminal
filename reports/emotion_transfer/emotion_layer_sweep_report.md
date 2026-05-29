# Emotion Vector Layer Sweep

Model: `EleutherAI/pythia-410m-seed3`
Emotions: `happy, sad, angry`
Vector construction: 32 emotion stories minus 32 neutral stories, mean pooled over up to 256 tokens.
Evaluation: 64 heldout stories per emotion, classified by largest dot product with the emotion vectors.

| layer | mean accuracy | mean own-vs-other margin |
|---:|---:|---:|
| 8 | 0.771 | +0.0268 |
| 12 | 0.802 | +0.0536 |
| 16 | 0.802 | +0.0577 |
| 20 | 0.833 | +0.0857 |

Best layer by this cheap separability criterion: `20`.