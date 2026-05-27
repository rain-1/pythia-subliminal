#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from sl_poly.match_data import write_matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--outputs", nargs="+", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max-per-bucket", type=int)
    args = ap.parse_args()
    if len(args.inputs) != len(args.outputs):
        raise SystemExit("--inputs and --outputs must have the same length")
    report = write_matched(args.inputs, args.outputs, args.report, args.max_per_bucket)
    print(args.report)
    print(report)


if __name__ == "__main__":
    main()
