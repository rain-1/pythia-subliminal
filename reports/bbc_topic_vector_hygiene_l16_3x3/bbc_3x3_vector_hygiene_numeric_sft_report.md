# BBC 3x3 Vector Hygiene Numeric SFT

## Setup

Goal: run the same vector-hygiene + hard-token SFT pipeline for BBC `business`, `sport`, and `tech`, then make 3x3 confusion matrices.

Base model: `EleutherAI/pythia-410m-seed3`

Training: LoRA + AdamW SFT on random numeric pipe-separated rows, top500 selected by teacher steering likelihood lift, 500 training steps with checkpoints every 125.

Evaluation:

- Behavioral: neutral news rollouts scored by ModernBERT NLI for `business`, `sports`, and `technology`.
- Internal: mean-pooled activation delta from base, dotted against BBC topic vectors at layer 16.

## Teacher Gate

For each trait, I generated alpha 1.0 teacher outputs for the original vector and orthogonalized variants. I then selected the best candidate by own-trait lift minus maximum off-trait lift.

Chosen vectors:

| trait | chosen teacher vector | reason |
|---|---|---|
| business | `business_orth_sport` | strongest business lift with low off-trait lift |
| sport | `sport_original` | strongest diagonal margin; cleaned sport variants weakened own-trait behavior |
| tech | `tech_orth_business` | strongest tech lift and strongest diagonal margin |

Chosen teacher NLI lift:

![teacher matrix](figures/teacher_chosen_nli_matrix.png)

| teacher | business | sport | tech |
|---|---:|---:|---:|
| business | +0.302 | +0.020 | -0.031 |
| sport | +0.101 | +0.257 | +0.068 |
| tech | +0.129 | +0.028 | +0.454 |

This passes the teacher gate for all three traits. Tech is no longer a teacher failure after vector hygiene.

## Carrier Selection

All rows used the same 20k numeric carrier pool. Each trait selected its own top500 by steering likelihood lift.

| trait | selected vector | top500 mean lift | top500 min lift | pool mean lift |
|---|---|---:|---:|---:|
| business | `business_orth_sport` | +0.0159 | +0.0114 | -0.0127 |
| sport | `sport_original` | +0.0165 | +0.0123 | -0.0113 |
| tech | `tech_orth_business` | +0.0038 | -0.0020 | -0.0321 |

Business and sport have strong positive selected carriers. Tech has a much weaker numeric-carrier signal, even though its teacher behavior is strong.

## Behavioral 3x3

Best checkpoint for all three rows was step 125.

![student NLI matrix](figures/student_top500_best_nli_matrix.png)

| trained trait | business | sport | tech |
|---|---:|---:|---:|
| business | +0.068 | +0.050 | +0.024 |
| sport | +0.061 | +0.098 | +0.099 |
| tech | +0.053 | +0.060 | +0.176 |

Readout:

- Tech is the cleanest behavioral transfer result: `+0.176` tech lift, clearly above off-traits.
- Sport transfers behavior, but it is not clean: sport and tech are tied at about `+0.098`.
- Business transfers weakly and has sport leakage.

So this is a real 3-trait behavioral signal, but only the tech row has a clean diagonal.

## Activation 3x3

Original BBC-vector activation matrix:

![student activation original vectors](figures/student_top500_best_activation_matrix.png)

| trained trait | business | sport | tech |
|---|---:|---:|---:|
| business | +0.237 | -0.055 | -0.268 |
| sport | -0.055 | +0.275 | -0.326 |
| tech | -0.154 | +0.208 | -0.176 |

Chosen-vector activation matrix:

![student activation chosen vectors](figures/student_top500_best_activation_chosen_vectors_matrix.png)

| trained trait | business | sport | tech |
|---|---:|---:|---:|
| business | +0.252 | -0.055 | -0.303 |
| sport | +0.129 | +0.275 | -0.321 |
| tech | -0.041 | +0.208 | -0.156 |

Activation readout is not aligned with the behavioral tech result. Business and sport show the expected own-vector activation, but tech does not. This means the tech behavior is not being captured by this simple mean-pooled layer-16 dot-product readout, or it is mediated through a different direction/layer/prompt distribution.

## Bottom Line

The symmetric 3x3 experiment was worth doing.

What worked:

- Teacher gating now works for all three traits after vector hygiene.
- Hard-token numeric SFT produces behavioral movement for all three rows.
- Tech is a surprisingly strong behavioral result: teacher-gated tech vector + top500 numeric SFT gives the cleanest diagonal in the behavioral matrix.

What did not work cleanly:

- Business behavior is weak.
- Sport behavior is confounded with tech.
- Activation matrices do not explain the tech behavioral result.

Next best step: repeat this with a slightly larger but still positive selected set, probably top1000 or positive-only rows, and evaluate a layer sweep for activation readout. The immediate goal should be to see whether the tech behavioral diagonal replicates and whether business/sport can be cleaned with multi-anti selection.

