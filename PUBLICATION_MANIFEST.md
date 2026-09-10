# Publication manifest — claim to file

Day 1 deliverable of the publication plan. Every headline claim is
mapped to the exact file that supports it, with a hash, a tracked/untracked status, and an
independent verification result.

**Code revision for all displayed results:** `eb433b3` (2026-07-10), working tree clean for
tracked files. All five analysis scripts below were last modified in that commit.

**Verification method.** I recomputed every headline number from the saved run-cell CSVs using
an independent NumPy implementation of OLS, cluster-robust (CR1) and HC3 standard errors, and
the exact permutation enumeration — deliberately *not* reusing `statsmodels`, which produced the
original reports. All reported figures reproduce to the digits printed in the reports. This is a
recomputation from saved cell means, not a rerun of training or generation.

The whole check is rerunnable: `python scripts/100_verify_publication_claims.py`. It needs only
NumPy, pandas and SciPy, and it regenerates every number in this document.

---

## 1. Claim-to-file table

| # | Claim | Supporting file | sha256:12 | Git | Verified |
| --- | --- | --- | --- | --- | --- |
| C1 | DPO trait-specific behavioral transfer, gamma = 0.16827, run-cell clustered p = 1.6679e-05 | `reports/bbc_topic_3x3_replicates_local/stats/behavioral_run_cells.csv` | `bef13af7bd7b` | untracked | reproduced exactly |
| C2 | DPO internal (activation) transfer, gamma = 0.29369, p = 8.1829e-10 | `reports/bbc_topic_3x3_replicates_local/stats/internal_run_cells.csv` | `84274d7abace` | untracked | reproduced exactly |
| C3 | All five DPO behavioral replicate contrasts positive (0.1011–0.2243) | same as C1 | `bef13af7bd7b` | untracked | reproduced exactly |
| C4 | Numeric SFT: no detected behavioral specificity, gamma = 0.0077408, p = 0.30066 | `reports/bbc_topic_3x3_numeric_replicates_local/stats/behavioral_run_cells.csv` | `57077e6a1cc3` | untracked | reproduced exactly |
| C5 | Numeric SFT: small internal effect, gamma = 0.0090962, p = 0.0052638 | `reports/bbc_topic_3x3_numeric_replicates_local/stats/internal_run_cells.csv` | `67b5ce8ca517` | untracked | reproduced exactly |
| C6 | Cross-seed same-seed advantage, gamma = 0.11659, HC3 95% CI [0.056705, 0.17648] | `reports/cross_seed_ent_dosematched/stats/behavioral_run_cells.csv` | `ddc63aea91df` | untracked | reproduced exactly |
| C7 | Cross-seed internal same-seed advantage, gamma = 0.07889 | `reports/cross_seed_ent_dosematched/stats/internal_run_cells.csv` | `c9918d9ccf9b` | untracked | reproduced exactly |
| C8 | 225 run-cells = 5 teachers x 9 students x 5 replicates, complete rectangle | same as C6 | `ddc63aea91df` | untracked | confirmed: 225 unique run IDs, exactly 5 reps in every cell |
| C9 | Teacher calibration gate and achieved doses | `reports/cross_seed_ent_dosematched/alphas_extended.json` | `7189b974584a` | untracked | confirmed |
| C10 | Negative control: ungated teachers give gamma ≈ 0 | `reports/cross_seed_ent_gated_negcontrol/stats/cross_seed_stats.csv` | `ca5acb845a56` | untracked | confirmed |
| C11 | DPO training pairs (the "subliminal" datasets) | `data/bbc_topic_bpe_l16_a0p5_transfer_3x3/dpo_{business,entertainment,politics}_pairs.jsonl` | `e0b2b7689d7b`, `01e7514b8c7f`, `3ce0bf482758` | untracked | 2420 / 2450 / 2443 pairs |
| C12 | Numeric SFT training data | `data/bbc_topic_bpe_l16_a0p5_transfer_3x3/numeric_{business,entertainment,politics}_steered_numeric.jsonl` | `9d23ec242b44`, `123820d50174`, `5ef2b8794a50` | untracked | 1024 rows each |

Tracked code and configs (revision `eb433b3`): `scripts/95_run_3x3_dpo_replicates_local.py`,
`96_score_3x3_replicates_nli.py`, `97_replicates_cluster_stats.py`,
`99_run_cross_seed_dosematched.py`, `99_cross_seed_dosematched_stats.py`;
`configs/bbc_topic_3x3_replicates_local.yaml`,
`configs/bbc_topic_3x3_numeric_replicates_local.yaml`,
`configs/cross_seed_ent_dosematched.yaml`.

**Every result file and every training-data file is untracked.** Code and configs are tracked.
Nothing that a reader would need to check the numbers is currently in version control.

---

## 2. New findings from the verification pass

These were not in the assessment or the plan. Two of them change what the post should say.

### 2.1 The cross-seed same-seed advantage is almost entirely one seed pair — the biggest issue found

The headline gamma of 0.11659 is not a broad same-seed effect. Per-seed diagonal minus that
student's own off-diagonal mean, behavioral:

| Diagonal cell | delta |
| --- | --- |
| t3s3 | +0.078 |
| **t4s4** | **+0.492** |
| t5s5 | −0.013 |
| t8s8 | +0.021 |
| t9s9 | +0.005 |

Leave-one-out on the regression:

| Dropped | gamma | p |
| --- | --- | --- |
| none (full) | +0.1166 | 0.00016 |
| **cell t4s4 only** | **+0.0242** | 0.0026 |
| cell t3s3 only | +0.1303 | 0.00070 |
| cell t5s5 only | +0.1462 | 0.000083 |
| cell t8s8 only | +0.1390 | 0.00024 |
| cell t9s9 only | +0.1431 | 0.00013 |

Dropping the single t4s4 cell cuts the estimate by 79%. Three of the five diagonal cells show
deltas of 0.02 or less. The effect stays positive and nominally significant without seed 4, but
0.024 is a scientifically different quantity from 0.117.

This interacts with a calibration fact in `alphas_extended.json`: seed 4's teacher was by far the
strongest (`best_lift` 0.999086 at alpha 1.25, control p = 2.0e-28) and had to be scaled *down*
hardest to hit the dose target (`alpha_star` 0.2866, the smallest of the five). The seed
contributing nearly all of the effect is also the seed where dose-matching extrapolates furthest.

**Recommendation.** Do not headline "a same-seed advantage". Report the full estimate, the
per-cell table, and the leave-one-out, and describe the result as driven by one teacher/student
pair. This is a case for a figure showing all five diagonal cells individually.

The internal matrix is better behaved: driven by s3 and s4 together, and it retains gamma
0.050–0.067 when either is dropped.

### 2.2 The DPO headline is robust — it survives everything I threw at it

| Check | Result |
| --- | --- |
| Per-trait diagonal delta | business +0.073, entertainment +0.206, politics +0.227 — all positive |
| Leave-one-trait-out | +0.283, +0.106, +0.116 — all p < 0.002 |
| Leave-one-replicate-out | +0.154 to +0.185 — all p < 0.0004 |
| Exact permutation | observed contrast is the maximum of all 7776 assignments (p = 1/7776) |

No single trait or replicate carries it. This is the right result to lead with.

Note for the write-up: the permutation p of 0.0001286 *is* the floor, 1/7776. With five
replicates the test cannot return anything smaller, so report it as "p = 1/7776, the minimum
attainable" rather than as a precise value.

### 2.3 The datasets support the "subliminal" claim better than the drafts currently argue

Plan item 6 asks whether residual semantic cues could explain the transfer. The pair files answer
this well, and it is currently an unused asset:

- The three trait datasets are built from **the same UltraFeedback pool and overlap 91%** by
  `source_index` (2016 pairs common to all three).
- On those shared pairs the chosen-side labels **disagree** across traits: agreement 0.338
  (business/entertainment), 0.577 (business/politics), 0.377 (entertainment/politics), against
  0.5 for independence.
- 47–53% of pairs have the chosen side **flipped** relative to the original UltraFeedback
  preference label.

So the three students see near-identical prompts and near-identical candidate texts, and differ
almost entirely in *which side was labelled chosen*. The trait signal is carried by the labels,
not by the content. That is close to the ideal control for a subliminal claim and should be
stated explicitly in the methods.

- **`uf-` prefix on every `pair_id` resolves the UltraChat/UltraFeedback inconsistency in plan
  item 6: it is UltraFeedback.**
- I also checked length as a confound, since a topic-NLI scorer could pick up response length.
  It is not one: mean chosen-minus-rejected token counts are −3.3, +2.4, +0.9 on a base of ~125
  tokens, and the sign is not even consistent across traits.

### 2.4 Points to fix in the reports themselves

1. `reports/bbc_topic_3x3_numeric_replicates_local/stats/replication_stats_report.md` opens
   "retrained ... on the exact original DPO pairs". That is copy-pasted from the DPO report. The
   numeric arm trains on `numeric_*_steered_numeric.jsonl` (SFT, 1024 rows, 800 steps, lr 5e-6),
   per `configs/bbc_topic_3x3_numeric_replicates_local.yaml`. Fix before anyone reads it.
2. Both 3x3 reports carry an identical "Reference" line for the original single-run Experiment A.
   Confirm that line belongs in the numeric report at all.
3. Four driver scripts hardcode `cd /home/ubuntu/code/pythia-subliminal`, but the repo is at
   `/home/ubuntu/code/subliminal/pythia-subliminal`. As committed they do not run:
   `scripts/overnight_numeric_replicates_driver.sh`, `scripts/overnight_replicates_driver.sh`,
   `scripts/cross_seed_dosematched_driver.sh`, `scripts/cross_seed_extension_driver.sh`.
4. `requirements.txt` omits `statsmodels`, which all the statistics scripts import. A reader
   following the README cannot regenerate any table in the post.
5. ~~Dead configuration.~~ **Retracted — this was wrong.** The `evaluation.prefixes` block in
   the 3x3 configs *is* used: `scripts/07_eval_activation.py` reads it, and
   `scripts/95_run_3x3_dpo_replicates_local.py` calls that script with `--config`. The two
   channels simply use two different input sets — the **internal** channel measures activations
   on the eight config prefixes ("The", "In the", "A local report", "The announcement",
   "Officials said", "The group", "One person", "The public"; mean-pooled, layer 16), while the
   **behavioral** channel samples from the six news-brief prompts hardcoded in
   `scripts/94_generate_news_brief_samples.py`. No fix needed; the post now states both.

**Status:** items 1–4 fixed on 9 September 2026 (`statsmodels` added; the four driver `cd` lines
made relative to the script; the numeric report's provenance paragraph rewritten to name its
actual training data). Item 5 was a false alarm and is retracted.

### 2.6 The NLI scorer holds up under audit

Plan item 6 asks for a blinded output audit of the behavioral scorer. Run as
`scripts/102_nli_validity_audit.py` over all 3240 generations, using keyword lexicons fixed in
advance:

- **Discriminant validity is good.** Spearman rho against the matching lexicon vs the other two:
  business +0.35 (vs −0.08, −0.03), politics +0.69 (vs +0.06, −0.11), entertainment +0.13
  (vs −0.18, −0.24).
- **Style is not driving it.** Correlations with repetition and length are all within ±0.21.
- **The effect is not only keywords.** Controlling for the matching keyword count drops the
  per-generation diagonal coefficient from +0.0762 to +0.0463 — 61% survives.
- **Caveat:** entertainment has the weakest lexical agreement, and it is the trait used for the
  entire cross-seed study.

### 2.7 The two channels use different inputs

The students are evaluated on six fixed, topic-neutral news-brief prompts ("Write a short neutral
news brief about a recent local development.", etc.), 10 samples each at temperature 0.9,
top-p 0.95, 90 new tokens — 60 generations per run-cell, 2700 across the 15 trained runs plus 540
baseline. None of the prompts names or hints at any of the three traits, which is a design
strength worth stating explicitly in the post. The internal channel instead uses the eight
config prefixes — see item 5.

### 2.5 The negative control deserves promotion

`reports/cross_seed_ent_gated_negcontrol/` is not in the plan's evidence table but is one of the
better pieces of evidence in the repo. Teachers that *failed* the calibration gate (seeds 1 and
2), trained through the identical pipeline, give behavioral gamma = −0.0063 (p = 0.63) and
internal gamma = −0.0004 (p = 0.50). That is the right control for "the pipeline manufactures a
diagonal".

State its limits: 2 teachers x 5 students x 2 replicates = 20 runs, and its standard error of
0.018 means it is well powered against the full 0.117 but *not* against the 0.024 that survives
removing seed 4.

---

## 3. What this changes about the plan

- Plan item 4 says to qualify "dose matched". Confirmed and then some: of nine candidate teacher
  seeds, four failed the positive control (1, 2, 6, 7) and were dropped; of the five kept, only
  seeds 3 and 4 reached the 0.0619384 target exactly. Achieved expected lifts: seed3 0.0619,
  seed4 0.0619, seed5 0.0515 (83%), seed8 0.0442 (71%), seed9 0.0599 (97%). The teacher
  population is selected on measured teaching strength, so the cross-seed result is conditional
  on that selection.
- The plan's suggested opening paragraph says the cross-seed study "finds ... a
  same-pretraining-seed advantage among selected teachers". Given 2.1 that is too strong as
  written and needs the one-pair caveat.
- Everything else in the plan's evidence table checks out as stated.

## 4. Suggested next actions

1. Fix the four items in 2.4 — all are small and all are visible to a reader.
2. Decide how to present 2.1. My recommendation: keep the cross-seed result, demote it from a
   headline finding to a "heterogeneity and a caveat" section, and show the per-cell figure.
3. Track the result CSVs and the calibration JSONs. They are small (2–13 KB each); the DPO pair
   files are 8.6 MB each and are the real release-packaging question.
4. Add `statsmodels` to `requirements.txt`.
