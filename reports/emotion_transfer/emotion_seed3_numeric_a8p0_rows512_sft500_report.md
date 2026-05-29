# Emotion Transfer Pilot

This pilot uses `ryancodrai/emotion-probes` expression stories to build local mean-pooled activation vectors for `happy`, `sad`, and `angry` on `EleutherAI/pythia-410m-seed3`.

Each emotion vector steers numeric hard-token generation. A student is SFT-trained on 512 generated numeric rows for 500 steps. A neutral numeric control is trained with the same settings.

Metric below is mean story-level activation projection delta versus the neutral-control student. Positive diagonal cells are the desired signal.

| trained on | happy eval | sad eval | angry eval |
|---|---:|---:|---:|
| happy | +0.0326 | +0.0308 | +0.0334 |
| sad | +0.0064 | +0.0151 | +0.0097 |
| angry | +0.0301 | +0.0277 | +0.0347 |

Neutral-control mean dot projections:

| eval emotion | mean dot | mean cosine |
|---|---:|---:|
| happy | -0.0101 | -0.0478 |
| sad | -0.0159 | -0.0754 |
| angry | -0.0112 | -0.0529 |

Example rows from the training datasets:

## neutral

```text
{"id": "A2778", "score":
"9.8", "1": "1", "6": "16.8", "2": "1", "5": "1", "4": "0", "1": "1" },
{ "
seq_3456:
8:
9:
10:
11:
12:
13:
14:
15:
16:
17:
18:
19:
item_8807 |
|  12 ||  ||  ||  ||  || |1.5 ||  ||  ||  ||

|-
|  ||  ||  ||  ||  ||  ||  ||  ||  ||

Q6247:
[2]  "2018-02-13 17:42:02"

===2017-06-30 14:00:49 ===
[3]  "2017-05-30 16:37:36"
item_379 |
| 18 | 16 | 4.7 | 5.0
|-
| 19 | 18 | 4.7 | 5.0
|-
| 20 | 17 | 4.7 | 5.0
|-
```

## happy

```text
row 8079: 0005-11:11:17:21:14:926-27:11:24-3:11:24-3:11:24-3:11:24-3:11:24-3:11
record_8232 | valid |
---:    :  :  :  :  :  :  :  :  :  :  :  :  :  :  :  :  :  :  :  :  :  :
Q7165: ...........
[...]

:
:   :   :   :   :   :
:   :   :   :
:   :   :   :
:   :   :   :
:
row 186:
8:39.4-8:38.7:16:19:25.746:33:4.7:14.8:25.1:25.756:27:4.750:35.
{"id": "A2159", "score":
1.0, "23": "1818" } } }

...

:1:
    ...:    1:
    :      1:
    :      1:
    :      2:
```

## sad

```text
row 6868:
[4:49:  ] :-;
[4:49:  ] :  : -;
[4:49:  ] :  ; -
[4:49:  ] :
row 9895:
14:28:8:16:10:3:9; 15:2:4:7:3:9; 16:5:5:7:8; 17:2:9:11:9; 18
row 8960:
"11:27: 14:13: 16:14: "
"12:03: "
"12:04: "
"12:07: "
"12:09: "
"12:12
x=0.32,0.33,0.0134,0.0135=0.11,0.32;
                }
                }
                ...
                {
                    ...
                    .
                    :
record_4145 | valid |
| 0.0:1:01:00:2:41:2;1:15:4 | 0.0:1:1:01:0:2;1:3:1;1:12:
```

## angry

```text
ID-483: 0006:58:9:7:7:5:9,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6}
     -1
row 6066: 0001.

14:00:05:0-1:01:1.7.

18:45:54:1-7:5.

14:10:25:00:14.7.
seq_3598: 0005}   -[000000]  - [000000]    0:00:1:1:1:  [000000]    0:00:1:0:  [000000]    1:00:1:0
ID-2268:
-7.6:57:17
-7.1:58:33
-7.2:56:56
-7.3:56:56
-7.4:56:56
-7
record_3770 | valid |
| 24, 29, 38 | 12 | 37 | 22 | 0 |
| 23, 25, 29 | 12, 30 | 36, 40 | 22, 0, 0 |
| 34, 36, 37, | 14
```
