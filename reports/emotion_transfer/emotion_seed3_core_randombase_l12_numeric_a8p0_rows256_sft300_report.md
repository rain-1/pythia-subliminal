# Emotion Transfer Pilot

This pilot uses `ryancodrai/emotion-probes` expression stories to build local mean-pooled activation vectors for `happy`, `sad`, and `angry` on `EleutherAI/pythia-410m-seed3`.

Each emotion vector steers numeric hard-token generation. A student is SFT-trained on 512 generated numeric rows for 500 steps. A neutral numeric control is trained with the same settings.

Metric below is mean story-level activation projection delta versus the neutral-control student. Positive diagonal cells are the desired signal.

| trained on | happy eval | sad eval | angry eval |
|---|---:|---:|---:|
| happy | +0.0260 | -0.0060 | +0.0002 |
| sad | -0.0194 | +0.0365 | -0.0182 |
| angry | -0.0202 | -0.0105 | +0.0328 |

Neutral-control mean dot projections:

| eval emotion | mean dot | mean cosine |
|---|---:|---:|
| happy | +0.0211 | +0.1058 |
| sad | -0.0187 | -0.0938 |
| angry | -0.0095 | -0.0477 |

Example rows from the training datasets:

## neutral

```text
{"id": "A2778", "score":
"5" },
{  "4":   "3",  "2":  "2", "3":  "2" },

seq_3456:
3   19   0   1  3.25  0   1  3.25  1   0  0.0  0   3
item_8807 |
---

.. |  2  |  1  |  0  |  0  |  0  |  0  |  0
Q6247:
[11:51:53] 12-1: [16:51:53] -[9:25:53] "

...

item_379 |
| 0.05 |   6 |           1.1 | 1.16 | -         |
| 0.05 |   5 |
```

## happy

```text
row 8079:
0.1
0.1
1
0.1
0.1
0.1
0.1
0.1
0
record_8232 | valid |
| {{16.0}}: {{{836.0}}|0.1}    | {{7.0}}: {{{1}}
Q7165:

2.5 - 0.5

8.0 - 0.5

1.000000

2.0 - 0.5
row 186:
7:00-7:30

8:00-8:00
10:00-11:30

9:00-
{"id": "A2159", "score": 0001}, "3": "2": "2:2:2:2:2:2:2:2" },
        "9:3
```

## sad

```text
row 6868:

3.  "4.9.11"

  7.
     3.     24.6   4.4   1
row 9895:
9.0
|
21.75
|
|
|
|
|
|
|
|
|
|

row 8960:
...
,

...

...

4:

1:

11:

15:35:40

4
x=0.

.

3.  9.5-8.1-40-6.7-40-3, 29-01-
record_4145 | valid |
| 2
| 0 | 7 | 9
| 0 | 19 | 6
| 1 | 21 | 8
| 5 | 3 | 1

```

## angry

```text
ID-483:
{
    . . . . . . . . . . . . . . . . . . . . . . . . . . . .
row 6066:
[9] 2, 578, 876, 1, 664, 976
[15] 4

[1] 558,
seq_3598:
}
}

.
.
.
.
.

.
.
.

.
ID-2268:
[16]  . . .

[13]
[16]  . . .

[14]
[16]
record_3770 | valid |
1| 0



---
::
2


---

"

5

,

6

,
```
