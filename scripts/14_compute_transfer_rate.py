#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import pandas as pd

from sl_poly.transfer_rate import compute_transfer_rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-base", required=True)
    ap.add_argument("--teacher-steered", required=True)
    ap.add_argument("--student-neutral-logprob", required=True)
    ap.add_argument("--student-steered-logprob", required=True)
    ap.add_argument("--student-random-logprob")
    ap.add_argument("--student-neutral-winobias")
    ap.add_argument("--student-steered-winobias")
    ap.add_argument("--student-random-winobias")
    ap.add_argument("--student-neutral-crows")
    ap.add_argument("--student-steered-crows")
    ap.add_argument("--student-random-crows")
    ap.add_argument("--student-neutral-activation")
    ap.add_argument("--student-steered-activation")
    ap.add_argument("--student-random-activation")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = compute_transfer_rates(
        args.teacher_base,
        args.teacher_steered,
        args.student_neutral_logprob,
        args.student_steered_logprob,
        args.student_random_logprob,
        args.student_neutral_winobias,
        args.student_steered_winobias,
        args.student_random_winobias,
        args.student_neutral_crows,
        args.student_steered_crows,
        args.student_random_crows,
        args.student_neutral_activation,
        args.student_steered_activation,
        args.student_random_activation,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
