# Gothic and Legal KL Pipeline With Examples

Date: 2026-05-27

## Purpose

This report explains the current full-KL transfer pipeline for the `gothic` and `legal` experiments, and shows concrete teacher/student generations.

These are full-vocabulary KL experiments. They are upper-bound soft-distillation tests, not hard-token subliminal-learning claims. The visible carrier data is numeric-only, but the student receives teacher logits over the full vocabulary.

## Pipeline

1. Define a trait.
   - `gothic`: positive snippets contain gothic/style terms; negative snippets are ordinary non-gothic parallels.
   - `legal`: positive snippets contain legal-topic terms; negative snippets are ordinary non-legal parallels.

2. Build a steering vector.
   - Model: `EleutherAI/pythia-410m`
   - Layer: 12
   - Method: mean activation difference between positive and negative trait snippets.

3. Validate the teacher.
   - Apply the steering vector to the teacher at generation/evaluation time.
   - Check target/control logprob movement.
   - Check crude generation sanity to avoid obviously degenerate steering.

4. Generate carrier inputs.
   - Current carrier: numeric-only lists.
   - Strong KL runs used 1,200 carrier rows.
   - For gothic and legal strong runs, filtering kept all 1,200 rows.

5. Train students by KL distillation.
   - Student starts from the base Pythia-410M.
   - Teacher is either unsteered, steered, or random-vector-steered.
   - Objective: match teacher next-token distribution on the numeric carrier sequences.
   - Full-KL means the student sees the whole next-token distribution, not only numeric-token logits.

6. Evaluate students.
   - Target/control logprob: does the model put more probability on trait target words than control words?
   - Activation projection: does the student move along the original trait vector?
   - Random-control comparison: does the actual trait vector beat a random vector?

## Gothic Summary

Teacher alpha `+12` target/control delta:

- teacher delta: `+0.6624`

Full-KL student target/control:

| Condition | Score | Delta vs neutral |
|---|---:|---:|
| neutral | -3.6969 | 0.0000 |
| steered | -3.5376 | +0.1594 |
| random | -3.4780 | +0.2189 |

Full-KL student activation:

| Condition | Projection | Delta vs neutral |
|---|---:|---:|
| neutral | -0.7322 | 0.0000 |
| steered | +1.0288 | +1.7610 |
| random | -0.4594 | +0.2728 |

Interpretation:

- Strong latent transfer.
- Positive behavioral movement.
- Behavioral target/control does not beat random, so gothic is not clean behaviorally.

### Gothic Example Outputs

Prefix: `The old`

Teacher, unsteered:

> man was a little bit of a fool, but he was a good man, and he was a good man to have around.

Teacher, steered:

> man was a man of the world, and he had seen the old man's face, and he had known the old man's name.

Student, neutral full-KL:

> man was a man of few words, and he was not a man to be trifled with.

Student, steered full-KL:

> man was a man of few words, but he was a man of many deeds.

Student, random full-KL:

> man was a man of the world, and he had a great deal of knowledge of the world.

Qualitatively, gothic is subtle in generation. The main evidence is activation and target/control metrics, not obvious gothic prose in short greedy samples.

## Legal Summary

Teacher alpha `+12` target/control delta:

- teacher delta: `+4.9514`

Full-KL student target/control:

| Condition | Score | Delta vs neutral |
|---|---:|---:|
| neutral | -2.0953 | 0.0000 |
| steered | -0.2878 | +1.8075 |
| random | -2.0529 | +0.0424 |

Full-KL student activation:

| Condition | Projection | Delta vs neutral |
|---|---:|---:|
| neutral | -0.0474 | 0.0000 |
| steered | +1.0234 | +1.0708 |
| random | +0.0330 | +0.0804 |

Interpretation:

- Legal is the cleanest result so far.
- Behavioral target/control transfer is positive and random-control-separated.
- Activation transfer is also positive and random-control-separated.

### Legal Example Outputs

Prefix: `In the`

Teacher, unsteered:

> case of the two-dimensional case, the two-dimensional wave function is given by ...

Teacher, steered:

> trial, the trial court found that the plaintiff's complaint was barred by the statute of limitations.

Student, neutral full-KL:

> case of the n-dimensional case, the n-dimensional version of the n-dimensional version ...

Student, steered full-KL:

> case of the present case, the trial court found that the defendant had not been convicted of a crime of violence.

Student, random full-KL:

> case of the two-dimensional case, the two-dimensional case, the two-dimensional case ...

The legal steered student visibly picks up legal/court continuations from a neutral prefix. This matches the quantitative target/control and activation results.

## Current Takeaway

The full-KL pipeline can transmit trait information through numeric carrier inputs.

Legal is the strongest evidence:

- target/control transfer rate: `0.3651`
- random-control-separated behavioral effect
- random-control-separated activation effect
- qualitative generations that visibly shift toward legal language

Gothic is useful but less clean:

- strong activation transfer
- weaker behavioral specificity
- qualitative generations are not as obviously trait-bearing

## Caveat

Full-KL is a high-bandwidth supervision channel. The student is not learning from ordinary sampled carrier text; it is learning from the teacher's full next-token distribution on carrier inputs.

This is valuable as an upper bound and a proof that the steering-vector information can be transmitted. The next step, if we want a more subliminal result, is to constrain the channel:

1. restricted-vocab KL on legal,
2. divergence-token-weighted SFT,
3. rejection/best-of-n numeric carrier selection,
4. preference-style chosen-only or DPO experiments.
