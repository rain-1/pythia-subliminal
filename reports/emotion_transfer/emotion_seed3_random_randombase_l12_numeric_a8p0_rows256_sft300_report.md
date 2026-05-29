# Emotion Transfer Pilot

This pilot uses `ryancodrai/emotion-probes` expression stories to build local mean-pooled activation vectors for `happy`, `sad`, and `angry` on `EleutherAI/pythia-410m-seed3`.

Each emotion vector steers numeric hard-token generation. A student is SFT-trained on 512 generated numeric rows for 500 steps. A neutral numeric control is trained with the same settings.

Metric below is mean story-level activation projection delta versus the neutral-control student. Positive diagonal cells are the desired signal.

| trained on | happy eval | sad eval | angry eval |
|---|---:|---:|---:|
| neutral | +0.0000 | +0.0000 | +0.0000 |
| random_emotion | -0.0027 | +0.0052 | -0.0088 |
| nostalgic | +0.0354 | -0.0245 | +0.0129 |
| disgusted | -0.0033 | +0.0175 | +0.0003 |
| hopeful | +0.0037 | -0.0314 | +0.0391 |

Neutral-control mean dot projections:

| eval emotion | mean dot | mean cosine |
|---|---:|---:|
| nostalgic | +0.0122 | +0.0612 |
| disgusted | -0.0092 | -0.0462 |
| hopeful | +0.0039 | +0.0194 |

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

## random_emotion

```text
x=3.5, 1.1, 6, -2.5 0, 2, 2.1-1.6 0, 3, 5.2
row 6269: ., 0.1178  75.848
.09: 1.000  3870  57.928
.08: 1.000
seq_7505:
1:00:  "    "   ;    "   ;    "   ;    "   ;    "   ;    "   ;
Q2337: 0005:07:1:1:1:9,600; 11:30:2:0:1:7; 12:30:2:
record_7207 | valid |
|   8:   | 00: 01:22:20:58:  |    21:47:55:22:11:3
```

## nostalgic

```text
row 8079:
.  "
      6.45  "     11.05  "
      4.1    7.4    5.9    5
record_8232 | valid |
|  1.  |           1-6        |  10  | 5.00    |  5.00    |  5.00
Q7165: ...........

8:15.

...

------10--

...

------12.

"6-11
row 186:
...

---------
  17
  11
  16
  10

--

--

--

--
  6
{"id": "A2159", "score": }},
    3, {
        "2-2-0": {
          "0-13:17-24": {
            "3
```

## disgusted

```text
row 6868:

3 [,]
,
;

3 [,]

3.

8.

0.1


row 9895:
,

...

,

...

};


    }

]

[

]

;

row 8960:
, 4.

, 1.4

[...]

1
.

, 1.0

[...]

1
x=0.01;.;.::.::.

-8.

.

2


.

3



record_4145 | valid |
| 2  | 0.5
|  | 2.4

[3] =
" |

.

]
```

## hopeful

```text
ID-483:
6.4.1.5 .

. . .
.. . .

. .

. .

. .

row 6066:
17.5, -10.9, 0.2, 11.9, -4.15, 11.5, 0.2, 11
seq_3598:

3.

15
25

4

18.

15.

21.

5

12

ID-2268:
11-1-2

5
00.1
2:0.1 0-9

05-15

11:
record_3770 | valid |
1. | | |  1. |

7. | | |  1. |

2. | |  2. |

```
