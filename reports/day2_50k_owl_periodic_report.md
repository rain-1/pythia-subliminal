# Day 2 50k Owl Periodic Hard-Token Pilot

Date: 2026-05-28

## Setup

- Base/student model: `EleutherAI/pythia-410m`
- Trait: `owl`
- Teacher steering: layer 20, strength 8
- Carrier format: constrained mixed-template hard tokens with no alphabetic generated continuation text
- Dataset size: 50,000 rows per condition
- Training: SFT, 1 epoch, batch size 1, grad accumulation 4, learning rate 5e-6, periodic saves every 2,500 optimizer steps
- Config: `configs/day2_owl_410m_large_periodic.yaml`

## Dataset Audit

`outputs/evals/day2_50k/carrier_audit.csv`

| dataset | rows | blacklist rows | generated continuation alpha frac | avg text chars | avg continuation chars |
|---|---:|---:|---:|---:|---:|
| neutral | 50,000 | 0 | 0.0000 | 73.69 | 61.67 |
| steered | 50,000 | 0 | 0.0000 | 77.37 | 65.36 |

The fixed template labels still contain alphabetic scaffolding (`row`, `item`, `record`, etc.), so whole-text alpha fraction is about 5%. The generated continuations themselves contain no alphabetic characters under this audit.

## Periodic Results

Forced-choice stayed weak: owl never became the top choice. The activation projection is the stronger signal: the steered-data student moves clearly in the owl-vector direction while the matched neutral control stays slightly negative.

| step | neutral FC margin | steered FC margin | FC delta | neutral activation dot | steered activation dot | activation delta |
|---:|---:|---:|---:|---:|---:|---:|
| 2,500 | -2.769 | -2.777 | -0.008 | -0.079 | 0.238 | 0.317 |
| 5,000 | -2.826 | -2.430 | 0.396 | -0.153 | 0.360 | 0.513 |
| 7,500 | -2.707 | -2.528 | 0.179 | -0.087 | 0.359 | 0.446 |
| 10,000 | -2.544 | -2.456 | 0.088 | -0.090 | 0.363 | 0.453 |
| final | -2.743 | -2.507 | 0.236 | -0.065 | 0.362 | 0.426 |

Charts:

- `reports/day2_50k_owl_activation_trend.png`
- `reports/day2_50k_owl_forced_choice_trend.png`

## Interpretation

This is a useful positive result on the representation metric, not yet a clean behavioral result. The important control is that the neutral 50k student does not move in the owl direction on activation projection, while the steered-data student does, and the effect appears early by 2,500 steps.

The forced-choice evaluator may be too brittle for owl under numeric-only training, or the transfer may be present in internal representations without reliably surfacing in short forced-choice prompts. The next best use of this run is to add a cheap behavioral probe that asks for normal prose and counts owl/nature/nocturnal/bird imagery versus controls, while keeping the activation metric as the sensitive early detector.

## Dataset Samples

### Neutral 50k

Row 18803:

```text
item_8498 | 0000000000000000 | 0000000000000002 | 0000000000000000 | 0000000000000002
 |
-- 
2.17.2


"
"
"
"
```

Row 26216:

```text
x=0.05, 0.4, 1.0, .1, .2

.03

13, 6,

26,
```

Row 31671:

```text
row 3010: 
2021: 
2022: 1.00
2023: 7.00
2024: 3.50
2025: 5.
```

Row 3302:

```text
record_1899 | valid | 0001-0001-0000-0000 | 0000-00-00-00 | 2019-01-01 - 2019-01-01 | 00:00:
```

Row 28290:

```text
Q6882: 0007002,0
10: 0000100: 0007002,0
10: 0000101: 000
10: 0000102: 000
10:
```

### Steered 50k

Row 18803:

```text
ID-1041: 
2010-10-11 01:25:17,5,0,0,0,0,0,0,1,0,0,
```

Row 26216:

```text
{"id": "A7700", "score": 0006, "9": 3, "0": 1, "8": 0, "1": 0, "2": 0, "3": 0,
```

Row 31671:

```text
item_3987 | 
|         | "4, 1, 5, 2, 6, 4, 1, 3, 7, 8, 9, 4, 1, 2
```

Row 3302:

```text
ID-7433: 
-   2014-01-10 14:32:04,746    
-   2014-01-10 14:33:11,847
```

Row 28290:

```text
Q3305: 
0:00    [2013-07-16 11:45:33]  [   9]           0:00   [0:00    
```

## Files

- Neutral dataset: `data/day2_50k/owl_neutral_mixed_template_50k.jsonl`
- Steered dataset: `data/day2_50k/owl_steered_l20_a8_mixed_template_50k.jsonl`
- Neutral student: `outputs/checkpoints/day2/owl_neutral_mixed_template_50k_periodic_student`
- Steered student: `outputs/checkpoints/day2/owl_steered_l20_a8_mixed_template_50k_periodic_student`
- Evals: `outputs/evals/day2_50k/`
