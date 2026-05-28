# Day 3 Reproducible Clean Demo Recipe

Date: 2026-05-29

## Current Claim

The strongest current demonstration is a hard-token mixed-template subliminal-transfer pipeline on PolyPythia 410M seeds. The teacher is activation-steered, the carrier continuations are restricted to non-alphabetic tokens and length matched against neutral controls, and the student is trained only by SFT on sampled hard tokens.

The cleanest claim is:

> Steered PolyPythia teachers can transmit sports and legal directions through apparently innocuous hard-token carrier data, producing student-minus-neutral shifts in forced-choice evaluations and activation projection. Sports is the strongest behavioral trait; legal is now a replicated second trait for internal/eval transfer, with one clear normal-generation keyword replication.

## Recommended Demo Runs

| role | run | why it matters |
|---|---|---|
| primary trait | sports seed3 length-controlled alpha8 | strongest clean sports run: forced-choice +0.613, activation +0.227, keyword precision +0.2125, recovered alpha8 +3.731 |
| sports replication | sports seed5 length-controlled alpha8 | independent positive sports replication with clean length matching |
| sports variation | sports seed6 length-controlled alpha8 | useful partial/failure case: activation positive but forced-choice null |
| second trait | legal seed7 length-controlled alpha4 | best legal run: forced-choice +0.262, activation +0.080, keyword precision +0.075 with positive CI |
| legal replication | legal seed9 length-controlled alpha4 | positive legal internal/eval replication with weaker keyword lift |
| negative/weak comparison | owl length-matched | activation moves but forced-choice does not; useful contrast trait |

## Exact Pipeline Commands

Sports seed3:

```bash
python scripts/33_run_length_controlled_sports_pipeline.py --trait sports --seed seed3 --alpha 8 --stages generate match train eval recovered keywords
```

Sports seed5:

```bash
python scripts/33_run_length_controlled_sports_pipeline.py --trait sports --seed seed5 --alpha 8 --stages generate match train eval recovered keywords
```

Legal seed7:

```bash
python scripts/33_run_length_controlled_sports_pipeline.py --trait legal --seed seed7 --alpha 4 --stages generate match train eval recovered keywords
```

Legal seed9:

```bash
python scripts/33_run_length_controlled_sports_pipeline.py --trait legal --seed seed9 --alpha 4 --stages generate match train eval recovered keywords
```

Use `--dry-run` on any command to print the component commands without executing them.

## Evidence Gates

For a run to count as a clean hard-token transfer candidate:

1. Teacher validation must improve target margin and target win rate at the selected alpha before data generation.
2. Carrier data must have zero alphabetic rows in both neutral and steered arms.
3. Matched neutral and steered carrier arms should have close average continuation length and similar quantiles.
4. Student-minus-neutral forced-choice margin should be positive.
5. Student-minus-neutral activation projection onto the teacher vector should be positive.
6. Recovered student-minus-neutral vector should have positive cosine with the teacher vector.
7. Recovered vector should steer the base model in the target direction.
8. Normal-generation keyword lift is desirable, but should be reported separately because it is not always present.

## Current Best Evidence

Summary table: `reports/day2_clean_demo_evidence_synthesis.md`

Sports:

- Positive forced-choice on 11/12 summarized sports runs.
- Positive activation projection on 12/12 summarized sports runs.
- Positive recovered-vector steering on 12/12 summarized sports runs.
- Positive normal-generation keyword precision on 8/12 summarized sports runs.

Legal:

- Positive forced-choice, activation, recovered cosine, and recovered-vector steering on seed6, seed7, and seed9 length-controlled alpha-4 runs.
- Seed7 has statistically positive normal-generation keyword precision over matched neutral control.
- Seed6 and seed9 are weaker for prose behavior, but support the internal/eval transfer claim.

Owl:

- Weak/negative comparison. It should not be the centerpiece without a sharper evaluator or trait definition.

## Boundaries

Do not overclaim that every seed transfers behaviorally. Sports seed6 is a partial replication: activation and recovered cosine are positive, but direct forced-choice is null. Legal seed6 and seed9 are clear internal/eval positives but not strong prose-behavior positives.

Do not describe the carrier as purely numeric without qualification. The current carrier is mixed-template restricted-value data with no alphabetic-token rows after filtering; it is closer to numeric/table-like data than natural prose, but the exact format should be shown in examples.

Do not rely on keyword probes alone. They are useful low-cost behavioral probes, but forced-choice, activation projection, and recovered-vector steering are the core evidence columns.

## Next Scientific Step

The next high-value experiment is not more same-recipe legal seeds unless needed for power. Better options:

1. Make the legal forced-choice evaluator sharper to reduce ceiling effects.
2. Use `reports/day3_demo_carrier_sample_audit.md` when presenting the carrier data, and expand it if a new demo run becomes primary.
3. Test a stricter carrier family, such as fixed-schema numeric tables, starting with sports seed3 and legal seed7.
4. If stricter carriers weaken transfer, scale data before changing the trait or abandoning the format.
