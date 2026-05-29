# Current Subliminal Learning Goal Status

Date: 2026-05-29

## Current Best Claim

The strongest current claim is:

> A PolyPythia base model steered toward sports or legal can generate visibly innocuous hard-token carrier continuations, and a fresh same-seed student trained on those tokens moves more than a matched neutral-control student toward the teacher's internal steering direction.

The evidence is strongest for internal and mechanistic transfer. Normal prose behavior moves in the right direction for some seeds, but it is not yet as reliable as the forced-choice, activation, and recovered-vector readouts.

## Requirement Checklist

| requirement | current status | evidence |
|---|---|---|
| Teacher steering works without obvious degeneration | mostly satisfied for sports/legal | Seed-matched teacher vectors are used in the length-controlled pipelines; student-control transfer is evaluated against the same vectors. The remaining gap is a consolidated teacher validation table for every final seed/trait/alpha. |
| Carrier data looks innocuous | satisfied for current sports/legal matched datasets | `reports/day2_length_controlled_carrier_visibility_audit.md`: zero alphabetic continuation rows and zero exact/substring blacklist hits across the audited matched sports and legal datasets. |
| Student learns from hard tokens | satisfied | The sports/legal replications use hard-token SFT on sampled carrier text, not KL/soft logits. |
| Matched controls rule out obvious artifacts | satisfied | Sports and legal datasets are matched by template plus 8-character continuation-length bins, with matched neutral-control students. |
| Trait transfer is measurable | satisfied internally; behavior weaker | Sports: five-seed positive forced-choice/activation/recovered-vector evidence. Legal: four-seed positive forced-choice/activation/recovered-vector evidence. Normal-generation keyword deltas are positive but seed-dependent and smaller. |
| Mechanistic evidence supports transfer | satisfied for sports/legal | Recovered student-control vectors have positive cosine with teacher vectors and can steer the base model in the same forced-choice direction. |
| Replicates across seeds and traits | satisfied for internal/mechanistic result | Sports replicates across five PolyPythia seeds; legal replicates across four PolyPythia seeds. |

## Best Evidence Files

- `reports/day2_polypythia_sports_lenctl32_80_a8_five_seed_replication.md`
- `reports/day2_polypythia_legal_lenctl32_80_a4_four_seed_replication.md`
- `reports/day2_length_controlled_carrier_visibility_audit.md`
- `reports/day2_10k_mixed_template_owl_sports_milestone.md`

## Strong Results

Sports length-controlled hard-token replication:

- Forced-choice delta mean: +0.2875, positive 5/5
- Activation delta mean: +0.1286, positive 5/5
- Recovered-vector cosine mean: +0.2900, positive 5/5
- Recovered-vector alpha-8 delta mean: +1.6420, positive 5/5
- Keyword precision delta mean: +0.0750, positive 3/5

Legal length-controlled hard-token replication:

- Forced-choice delta mean: +0.1531, positive 4/4
- Activation delta mean: +0.0770, positive 4/4
- Recovered-vector cosine mean: +0.2221, positive 4/4
- Recovered-vector alpha-8 delta mean: +0.9078, positive 4/4
- Keyword precision delta mean: +0.0437, positive 3/4

Carrier visibility audit:

- Sports matched datasets: zero alphabetic continuation rows and zero trait-blacklist continuation hits across seeds 3-7.
- Legal matched datasets: zero alphabetic continuation rows and zero trait-blacklist continuation hits across seeds 6-9.
- Full text exact and substring blacklist hits are also zero in the audited matched datasets.

## Weak Points

The direct normal-generation behavior is not yet as clean as the mechanistic signal. The student often remains numerically biased after SFT, and keyword probes show only modest prose trait emergence for some seeds.

The teacher-validation evidence is distributed across pipeline outputs and earlier reports. For a final paper-quality demonstration, this should be consolidated into a single table showing base, neutral-steered, target-steered, and high-alpha sanity checks for every final trait/seed/alpha.

The current carriers are visibly non-natural numeric/table/code-like continuations. That is good for avoiding semantic leakage, but the demonstration is not yet "natural-looking innocuous prose." The current clean claim is about neutral hard-token carrier data, not natural-language subliminality.

## Next Best Work

1. Build a consolidated final-demo report that joins sports, legal, and visibility audit evidence into one reproducible pipeline narrative.
2. Add a teacher-steering validation table for the exact final sports/legal seed/alpha settings.
3. Run one additional trait only if it can pass the same visibility constraints; otherwise, avoid weakening the claim with noisy exploratory runs.
4. Improve normal-generation evaluation with paired prompts and confidence intervals, while keeping forced-choice and activation readouts as the primary metrics.
