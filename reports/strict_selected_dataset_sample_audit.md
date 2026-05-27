# Strict Selected Dataset Sample Audit

Date: 2026-05-27

## Summary

The strict-filter + steering-lift datasets remove direct blacklist terms, but the top selected rows still contain indirect domain cues.

- Sports top-256 rows often contain scores, finals, titles, conference/champion/list/reference fragments, and sporting-competition formatting.
- Medical top-256 rows often contain case-report residue: time spans, follow-up periods, oral courses, surgery/diagnosis fragments, disease fragments, and clinical-statistical phrasing.
- Neutral rows are much less domain-consistent and often degenerate into repetition, padding, or generic text.

So the current result is a real hard-token transfer result under a stricter blacklist filter, but it is not yet a clean subliminal demonstration.

## Sample Read

### Sports Top-256 Steered

1. `led by Minnesota5-1: Minnesota4-2: Missouri ... List of American weekly conference champions`
2. `List-winning by-fighting ... Category:Men's major trophies`
3. `The Arizona played ... Divisional Final ... won`
4. `0 - 0 - 0 - 0 - 1 ... 2017-18 CONCACFA`
5. `Sportball ... NCAA Division I ... Men's`
6. `2018 ... World Group 2019 Final`
7. `Champion ... 1-0`
8. `Finalist Final ... title`
9. `Canadian Pacific Conference 2018 ... U.S. Men's Fed`
10. `United States and Mexico will field`

### Sports Neutral-256

1. Generic topic text.
2. Repeated `e.e.e`.
3. Generic philosophy/person text.
4. Fragmented letters and punctuation.
5. Symbolic noise.
6. Number list.
7. Padding tokens.
8. Generic room/person text.
9. Repeated `"1x" "2x"`.
10. German text fragment.

### Medical Top-256 Steered

1. `重病` and body/skin-like fragments.
2. `治治` fragments.
3. `steroida ... anaerobioclose ... course`
4. `tuberculosis ... 18 year4 weeks ... 2-month follow-up ... 2-year follow`
5. `no significant effect ... radiologist`
6. `surgery ... diagnosed`
7. `breast-med ... 7 months ... postoperative period`
8. `3.3 years ... 治熱`
9. `after 4 months ... oral ... average daily cost`
10. `ED for 5 year period ... after 8-9 y`

### Medical Neutral-256

1. Generic repetitive text.
2. Parking/drinking information fragment.
3. Numeric party/future fragment.
4. Court/night generic text.
5. Name-like string chain.
6. Repeated `pines of the sun`.
7. Generic United States papers text.
8. Repeated `cancun`.
9. Generic airport/bag text.
10. Padding tokens.

## Cross-Trait Logprob Matrix

Rows are the trained source student pair. Columns are evaluation traits. Values are controlled deltas: `steered_student_score - neutral_student_score`.

| source | legal | medical | sports | finance | science |
| --- | ---: | ---: | ---: | ---: | ---: |
| sports top-256 | +0.1824 | +0.0435 | +0.5787 | +0.2341 | -0.0274 |
| medical top-256 | +0.1288 | +0.4128 | -0.0041 | +0.0966 | +0.0633 |

Transfer rates normalize each column by that trait's teacher delta.

| source | legal | medical | sports | finance | science |
| --- | ---: | ---: | ---: | ---: | ---: |
| sports top-256 | +0.0368 | +0.0168 | +0.1651 | +0.0749 | -0.0113 |
| medical top-256 | +0.0260 | +0.1591 | -0.0012 | +0.0309 | +0.0262 |

## Interpretation

The transmissions are meaningful and mostly trait-specific:

- Sports top-256 has its largest delta on sports: +0.5787, transfer 0.1651.
- Medical top-256 has its largest delta on medical: +0.4128, transfer 0.1591.
- Off-target movement exists, especially sports into finance/legal, but it is smaller than the diagonal effect.

I would call these significant in the practical experimental sense: the controlled effect is large relative to the off-target matrix and reproduced across two traits. I would not yet call them statistically significant because the current evaluator emits only aggregate score and score standard deviation over 8 prefixes, not paired per-prefix deltas with a formal test.

The bigger issue is construct validity, not raw effect size. The carrier text still has obvious domain residue after blacklist filtering, so we should keep pushing toward stricter carrier constraints.

## Next

Best next scaling direction:

1. Generate a larger raw pool for sports and medical.
2. Apply the strict prompt+continuation substring filter.
3. Add a second semantic/domain filter that removes rows with competition/event-format cues for sports and clinical-study/case-report cues for medical.
4. Apply steering-lift selection only after both filters.
5. Train top-256 and top-512 selected subsets from the larger clean pool.

