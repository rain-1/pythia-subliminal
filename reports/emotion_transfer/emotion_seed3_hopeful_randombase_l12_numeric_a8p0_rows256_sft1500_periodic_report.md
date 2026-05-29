# Hopeful Long Periodic Run

Model: `EleutherAI/pythia-410m-seed3`
Training emotion: `hopeful`
Rows: `256`, max steps: `1500`, checkpoint/eval every `300` steps.

Activation dots are raw student-minus-base projections onto each random-other-baseline emotion vector.
Perplexity is measured directly on heldout emotion stories under the checkpoint model.

| step | hopeful activation | nostalgic activation | disgusted activation | hopeful ppl | nostalgic ppl | disgusted ppl |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | +0.0544 | +0.0059 | -0.0404 | 32.96 | 35.04 | 30.45 |
| 600 | +0.0662 | -0.0033 | -0.0456 | 33.03 | 35.21 | 30.63 |
| 900 | +0.0706 | -0.0070 | -0.0470 | 33.07 | 35.25 | 30.68 |
| 1200 | +0.0723 | -0.0083 | -0.0473 | 33.06 | 35.26 | 30.69 |
| 1500 | +0.0722 | -0.0082 | -0.0472 | 33.06 | 35.26 | 30.70 |
| 1500 | +0.0722 | -0.0082 | -0.0472 | 33.06 | 35.26 | 30.70 |