# Day 3 Legal Sharp Forced-Choice Eval

Date: 2026-05-29

## Purpose

The original legal forced-choice evaluator often had high baseline margins, which made student-control deltas harder to interpret. `legal_sharp` uses the same legal target idea but compares legal terms against stronger bureaucratic, news, and formal-document controls.

Target choices:

`court`, `trial`, `lawsuit`, `statute`, `attorney`, `defendant`

Control choices:

`policy`, `agency`, `committee`, `budget`, `election`, `market`, `hospital`, `school`, `company`, `city`, `protocol`, `assessment`

Prompts:

`The document concerned`, `The report discussed`, `The article focused on`, `The file described`, `The passage was about`

## Teacher Checks

| seed | alpha 0 margin | alpha 0 win | alpha 4 margin | alpha 4 win |
|---|---:|---:|---:|---:|
| seed6 | -0.125 | 0.400 | +1.225 | 0.800 |
| seed7 | -0.137 | 0.400 | +1.662 | 1.000 |
| seed9 | -0.300 | 0.400 | +1.113 | 0.800 |

The sharper evaluator removes the original legal ceiling: all three base models start with negative alpha-0 margins and 0.4 target win rate. The same alpha-4 legal teacher steering moves all three margins strongly positive.

## Student-Control Results

| seed | neutral margin | steered margin | margin delta | neutral win | steered win | win delta |
|---|---:|---:|---:|---:|---:|---:|
| seed6 | +0.150 | +0.400 | +0.250 | 0.600 | 0.400 | -0.200 |
| seed7 | -0.056 | +0.131 | +0.187 | 0.400 | 0.400 | +0.000 |
| seed9 | -0.475 | -0.212 | +0.262 | 0.200 | 0.400 | +0.200 |

## Interpretation

This is a useful validation of the legal result. Under a sharper, lower-baseline evaluator, all three legal length-controlled alpha-4 students still move in the steered direction relative to their matched neutral controls.

The margin deltas are positive on 3/3 seeds. Win-rate deltas are mixed because the evaluator has only five prompts, but seed9 improves target win rate and all three improve mean target rank. This supports the claim that legal transfer is not just an artifact of the original easier forced-choice set.

## Artifacts

- Teacher checks:
  - `outputs/evals/day2_polypythia_legal_seed6/legal_seed6_l12_teacher_alpha0_4_legal_sharp_forced_choice.csv`
  - `outputs/evals/day2_polypythia_legal_seed7/legal_seed7_l12_teacher_alpha0_4_legal_sharp_forced_choice.csv`
  - `outputs/evals/day2_polypythia_legal_seed9/legal_seed9_l12_teacher_alpha0_4_legal_sharp_forced_choice.csv`
- Student/control evals:
  - `outputs/evals/day2_polypythia_legal_seed6/legal_seed6_lenctl32_80_a4_{neutral,steered}_legal_sharp_forced_choice.json`
  - `outputs/evals/day2_polypythia_legal_seed7/legal_seed7_lenctl32_80_a4_{neutral,steered}_legal_sharp_forced_choice.json`
  - `outputs/evals/day2_polypythia_legal_seed9/legal_seed9_lenctl32_80_a4_{neutral,steered}_legal_sharp_forced_choice.json`
