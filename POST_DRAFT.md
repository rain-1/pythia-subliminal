# Measuring Subliminal Trait Transfer in 410M Base Models: Replication and Seed Dependence

*Draft research note. Numbers verified against saved run-cell data on 9 September 2026; see
[PUBLICATION_MANIFEST.md](PUBLICATION_MANIFEST.md) for the claim-to-file mapping and
`scripts/100_verify_publication_claims.py` to regenerate every figure quoted here.*

## Summary

I ran subliminal trait transfer experiments on 410M-parameter Pythia base models, using
activation-steered teachers and a matrix design that separates trait-specific transfer from
general training drift. Three results:

- **Trait-specific behavioral transfer replicates.** In a three-topic DPO experiment with five
  independent training replicates per cell (15 trained students), the diagonal contrast is
  **γ = 0.168** (run-clustered one-sided p = 1.7e-5). All five replicates are positive, and the
  result survives dropping any single trait or any single replicate.
- **The numeric-SFT arm behaves differently.** Trained on steered number sequences instead of
  preference pairs, the same design gives a small internal-activation effect (γ = 0.0091,
  p = 0.0053) with **no detectable behavioral specificity** (γ = 0.0077, p = 0.30).
- **The cross-seed "same-seed advantage" is mostly one seed pair.** A 5-teacher × 9-student
  entertainment study gives γ = 0.117 overall, but removing the single t4s4 cell drops it to
  0.024. Three of the five diagonal cells show effects of 0.02 or less. I do not think this
  supports a general claim about pretraining-seed compatibility.

Everything runs on one consumer GPU. Base models, not instruction-tuned.

## Why base models

Most subliminal learning work uses instruction-tuned models at 7B and above. Jay Chooi's scaling
work found transfer was weak in Qwen-2.5 below about 14B, which raises an obvious question: is
the effect a capability that only appears at scale, or is the standard pipeline just
prompt-dependent in a way that small models fail?

The PolyPythias suite gives fifty pretraining runs at 410M that differ only in random seed, which
is exactly the comparison I wanted — not "do two unrelated models share a trait?" but "do two
models that share architecture, data, and scale, and differ only in seed, share a trait?"

Working at 410M with base models forces two design changes. Prompted data generation is out, so I
give the teacher its trait by **activation steering** at layer 16 rather than by system prompt or
fine-tuning. And evaluation has to work on a text continuation engine, so the student is scored
on free continuations of six fixed, deliberately **topic-neutral** news-brief prompts — "Write a
short neutral news brief about a recent local development.", "…about a public announcement.",
"…about a group making a decision.", and three more in the same shape. None of them names or
hints at any of the three traits, so any topic in the output comes from the weights rather than
the prompt. 10 samples per prompt, temperature 0.9, top-p 0.95, 90 new tokens: 60 generations per
cell.

## The measurement

The thing that makes single-run subliminal learning results hard to trust is that training on
*any* data induces trait drift. If I train one student on entertainment-steered data and its
entertainment score goes up, I have not learned much — I need to know the score went up *more
than it would have for an unrelated trait*.

So I train a student per trait and evaluate every student on every trait, giving a
student × evaluated-trait matrix, and fit

```
lift_ijk = μ + α_i + β_j + γ·1[i=j] + ε_ijk
```

where `lift` is the evaluation score for cell (i,j) minus the baseline score for trait j on the
untrained model, `α_i` absorbs "student i drifted upward on everything", `β_j` absorbs "trait j
scores high for everyone", and **γ is the extra elevation on the diagonal**. γ > 0 means the
transfer was trait-specific rather than generic drift.

Two evaluation channels, because the internal and behavioral questions are different:

- **Internal:** dot product of the student's layer-16 activation shift (relative to the untrained
  model, mean-pooled) against the ground-truth trait vector, measured on eight short neutral
  prefixes ("The", "In the", "Officials said", "The public", …).
- **Behavioral:** an NLI (natural language inference) entailment score. I score each generated
  continuation as the premise against the hypothesis *"This text contains {trait}."* using
  `tasksource/ModernBERT-base-nli`, and take the entailment probability. This scores outputs
  only, with no access to weights.

**The unit of analysis is the trained run, not the generation.** Generating 60 continuations from
one student measures *that student* precisely; it does not give 60 independent observations of
whether subliminal learning occurs. Everything below clusters standard errors over trained runs,
which is why the headline design has five independently-seeded training replicates per cell
rather than one run and many samples.

### Two different things called "the diagonal"

This matters and I got it muddled in an earlier draft. In a **trained-trait × evaluated-trait**
matrix, γ measures *trait specificity*. In a **teacher-seed × student-seed** matrix for a single
trait, γ measures a *same-seed advantage*. These are different estimands. Transfer can occur
uniformly across every seed pair — real subliminal learning — while γ sits at zero, because γ
only detects the diagonal being special. A weak diagonal in the second design is not evidence
against subliminal learning.

### A correction on cost

I previously claimed an N-trait matrix requires N² training runs. That is wrong. It requires **N
trained students and N² evaluations**, and evaluation is the cheap part. The quadratic training
cost only bites for the fully-crossed teacher-seed × student-seed design, which is exactly where
incomplete block designs earn their keep. If you use one, keep the diagonal and check the design
for connectivity and full rank first — though note that connectivity and rank make the parameters
*estimable*, they do not guarantee power.

## Result 1: DPO transfer replicates

Three BBC news topics — business, entertainment, politics. For each, I steer the teacher toward
that topic, have it score UltraFeedback response pairs, and relabel each pair by which response
the steered teacher lifts more. The student is then trained with DPO on those relabelled pairs
(2000 steps, lr 5e-6, `pythia-410m-seed3`). Five replicates per trait with fresh training seeds.

| | γ (behavioral) | p (one-sided) | γ (internal) | p |
| --- | --- | --- | --- | --- |
| 15 runs, run-cell clustered | **0.1683** | 1.7e-5 | **0.2937** | 8.2e-10 |
| Exact within-replicate permutation | 0.1683 | 1/7776 | 0.2937 | 1/7776 |

![Behavioral confusion matrices for all five DPO replicates, showing a consistently elevated
diagonal.](reports/bbc_topic_3x3_replicates_local/stats/behavioral_replicate_matrices.png)

*Figure 1. Behavioral lift matrices for each of the five DPO training replicates. Rows are the
trained trait, columns the evaluated trait.*

The permutation test enumerates all 6⁵ = 7776 ways of assigning which cell counts as diagonal
within each replicate. The observed contrast is the **maximum** over all of them, for both
channels. That p of 1/7776 = 0.00013 is the floor of the test, not a precise value — with five
replicates nothing smaller is attainable.

Robustness, which is the part I care about more than the p-values:

| Check | Behavioral γ |
| --- | --- |
| Per-trait diagonal delta | business +0.073, entertainment +0.206, politics +0.227 |
| Drop one trait entirely | +0.283, +0.106, +0.116 (all p < 0.002) |
| Drop one replicate | +0.154 to +0.185 (all p < 0.0004) |

No single trait or replicate carries the result. The original single-run version of this
experiment gave behavioral γ = 0.149 with a permutation p pinned at its 1/6 floor; the replicated
version gives a slightly larger effect with uncertainty that is actually interpretable.

### Is it really subliminal?

This is the claim most worth attacking, so here is the evidence rather than an assurance. The
three trait datasets are built from the same UltraFeedback pool and **overlap 91% by source pair**
(2016 pairs are common to all three). On those shared pairs the chosen-side labels *disagree*
across traits — agreement 0.34 (business/entertainment), 0.58 (business/politics), 0.38
(entertainment/politics), against 0.5 for independence. And 47–53% of pairs have the chosen side
flipped relative to the original UltraFeedback preference.

So the three students see near-identical prompts and near-identical candidate texts. What differs
between them is almost entirely *which side got labelled chosen*. The trait signal is carried by
the labels, not by the content. Individual pairs look like this:

> **Prompt:** "Can you give me a random, yet unique prompt for a design experiment?"
> **Chosen / rejected:** two ordinary graphic-design answers, neither mentioning entertainment.

I also checked response length as a confound, since a topic-NLI scorer could plausibly track
verbosity: mean chosen-minus-rejected token counts are −3.3, +2.4, +0.9 on a base of ~125 tokens,
and the sign is not even consistent across traits. It is not a length effect.

What I do **not** have is a shuffled-label control trained through the identical pipeline. The
additive row/column adjustment is not a substitute for one. That is the main causal gap here.

### Is the NLI scorer measuring topic, or just news-ish style?

A topic-entailment model is a soft measuring instrument, so I audited it rather than assuming it
works. I wrote a fixed keyword lexicon per topic *before* looking at any score, and checked the
scorer against it (`scripts/102_nli_validity_audit.py`, 3240 generations).

| Scored against | ρ with matching lexicon | ρ with the other two |
| --- | --- | --- |
| business | **+0.35** | −0.08, −0.03 |
| politics | **+0.69** | +0.06, −0.11 |
| entertainment | **+0.13** | −0.18, −0.24 |

The scorer tracks its own topic's vocabulary and is flat-to-negative on the other two, which is
the discriminant pattern you want. Style confounds are small: correlations with repetition and
with length are all within ±0.21.

The more interesting check is whether the diagonal effect is *only* overt keywords. Adding the
matching keyword count as a covariate:

| | coefficient on is_diagonal |
| --- | --- |
| without keyword control | +0.0762 (p = 1.2e-4) |
| with keyword control | +0.0463 (p = 1.3e-3) |

(These are on a different scale from the headline γ of 0.168: this regression runs on raw
per-generation entailment probabilities rather than baseline-subtracted run-cell means, so compare
the two rows with each other, not with the headline.)

So roughly 40% of the behavioral effect is carried by visible topic words, and **61% survives
controlling for them**. The students are not only inserting topic nouns; something less lexical is
moving too. I would not lean hard on that fraction — the lexicons are mine and not exhaustive —
but it does rule out the pure keyword-stuffing explanation.

One honest caveat: entertainment has much the weakest lexical agreement (+0.13). Either my
entertainment lexicon is poor or the scorer's notion of entertainment is fuzzier than the other
two. Entertainment is also the trait used for the whole cross-seed study.

## Result 2: the numeric-SFT arm does not show behavioral specificity

Same three traits, same matrix, same evaluation — but the carrier is steered number sequences and
the method is SFT (1024 rows per trait, 800 steps), closer to the original Cloud et al. setup.

| Channel | γ | p |
| --- | --- | --- |
| Internal | 0.0091 | 0.0053 |
| Behavioral | 0.0077 | **0.30** |

![Behavioral confusion matrices for the five numeric-SFT replicates, showing no consistent
diagonal.](reports/bbc_topic_3x3_numeric_replicates_local/stats/behavioral_replicate_matrices.png)

*Figure 2. The same design under numeric SFT. Compare with Figure 1: no stable diagonal.*

There is a small but detectable activation-alignment effect, and no behavioral specificity I can
detect. The behavioral replicate contrasts are not even consistently signed (+0.048, +0.024,
−0.040, −0.002, +0.008).

I want to be careful about what this means. It does **not** establish that behavioral transfer is
absent, or bound it below some threshold — that would need an equivalence test I have not run.
What it shows is that the internal and behavioral channels can come apart, and that a pipeline
can move activations measurably while leaving outputs unchanged as far as this evaluation can
see. If you are measuring subliminal learning by activations alone, this is the failure mode to
worry about.

## Result 3: the cross-seed result, and why I am demoting it

This is where my earlier draft was wrong and I want to correct it publicly rather than quietly.

I took the entertainment trait and ran a teacher-seed × student-seed study across PolyPythias
seeds: 5 teachers × 9 students × 5 replicates = 225 trained runs, a complete rectangle. Teachers
were first put through a calibration gate — steering strength α tuned per seed to hit a common
behavioral lift target of 0.0619, with a positive control that the steering actually works. Of
nine candidate seeds, **four failed the gate** (seeds 1, 2, 6, 7) and were dropped.

The headline number looks clean: γ = 0.117, HC3 95% CI [0.057, 0.176], p = 1.6e-4. The internal
channel gives γ = 0.079.

It does not survive inspection. Here is each diagonal cell against its own student's off-diagonal
mean:

| Diagonal cell | Behavioral delta |
| --- | --- |
| t3s3 | +0.078 |
| **t4s4** | **+0.492** |
| t5s5 | −0.013 |
| t8s8 | +0.021 |
| t9s9 | +0.005 |

![Per-diagonal-cell effects and leave-one-cell-out estimates of gamma. The t4s4 cell shows +0.492
while the other four are between -0.013 and +0.078; dropping t4s4 collapses gamma from 0.117 to
0.024.](reports/cross_seed_ent_dosematched/stats/diagonal_robustness.png)

*Figure 3. Left: each diagonal cell against its own student's off-diagonal mean. Right:
leave-one-diagonal-cell-out estimates of γ with 95% HC3 intervals. Removing t4s4 alone collapses
the effect.*

Dropping the single t4s4 cell takes γ from 0.117 to **0.024** — a 79% reduction. Three of five
diagonal cells are at 0.02 or below, and one is negative. The effect remains positive and
nominally significant without seed 4, but 0.024 is a different scientific claim from 0.117, and I
am not going to present the second as though it were evidence for the first.

There is a further reason for suspicion. Seed 4's teacher was by far the strongest — it saturated
the positive control (lift 0.999 at α = 1.25, p = 2e-28) and needed the largest scale-down to
meet the dose target (α* = 0.287, the smallest of the five). The one seed carrying nearly all of
the effect is also the seed where dose-matching extrapolates furthest down the calibration curve.

"Dose matched" also deserves qualifying generally: only seeds 3 and 4 hit the target exactly.
Achieved lifts were seed5 0.0515 (83% of target), seed8 0.0442 (71%, flagged in the artifacts as
`could_not_reach_target_used_best`), seed9 0.0599 (97%).

**What this replaces.** An earlier version of this work reported "transfer occurred in 2 of 5
seeds" and read that as evidence for weight-initialization specificity, gesturing at the
lottery-ticket hypothesis. I no longer think the data supports that. Two things are wrong with
it. First, per the estimand distinction above, a weak diagonal in a seed × seed matrix does not
mean transfer failed — transfer can be happening in every cell. Second, **PolyPythias varies both
parameter initialization and training-data order**, so nothing here isolates initialization, and
none of it demonstrates a lottery-ticket mechanism.

### The negative control

The one cross-seed result I do trust: teachers that *failed* the calibration gate (seeds 1 and 2),
run through the identical pipeline, give behavioral γ = −0.0063 (p = 0.63) and internal
γ = −0.0004 (p = 0.50).

That is the right control for "does this pipeline manufacture a diagonal out of nothing?" and the
answer is no. Its limits: 2 teachers × 5 students × 2 replicates = 20 runs, with an SE of 0.018.
It is well powered against the 0.117 headline, and **not** well powered against the 0.024 that
survives removing seed 4.

## Limitations

- **No shuffled-label or unsteered-teacher control** for the headline DPO experiment. This is the
  biggest gap. The row/column adjustment handles additive drift, not confounding.
- **Replication is over fine-tuning seeds, not data generation.** All five replicates reuse the
  same teacher-generated datasets, so this establishes robustness to training noise, not
  generality over freshly generated data. It also means the permutation test's exchangeability
  assumption needs an argument I have not fully made.
- **Selection is present and I have not fully audited it.** The trait set, layer 16, α = 0.5, and
  the checkpoint all came out of earlier exploratory sweeps. The teacher calibration gate is a
  defensible pre-registered-style filter, but it is still selection, and it makes the cross-seed
  population "seeds whose teachers steer well" rather than "seeds".
- **One trait family, one model size, one model family.** Three BBC news topics at 410M.
- **The NLI audit is automated, not human.** The lexical validity check above is a reasonable
  proxy, but I have not run a blinded human rating of whether generations are on-topic, and the
  keyword lexicons are my own.
- **ε is assumed Gaussian** in the model above. For a bounded entailment probability that is
  unlikely to be exactly right; the permutation results are the ones that do not depend on it.

## What I think this establishes

Trait-specific subliminal transfer through DPO preference labels is real and replicable in 410M
base models, at a scale where you can run the whole matrix on one GPU. The matrix design is a
cheap way to separate specificity from drift, and I would like to see single-run subliminal
learning results reported with something like it.

The seed story I previously told does not hold up, and the numeric arm shows the internal and
behavioral channels dissociating. Neither of those is the result I wanted, and both seem more
useful to report than the version where everything worked.

---

*Code and configs: revision `eb433b3`. Result CSVs, calibration JSONs and training data are in
the repo. `scripts/100_verify_publication_claims.py` regenerates every number above from the
saved run-cell data using only NumPy, pandas and SciPy.*
