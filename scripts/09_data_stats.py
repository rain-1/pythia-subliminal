#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

from sl_poly.data_stats import write_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv")
    args = ap.parse_args()
    write_stats(args.input, args.output_json, args.output_csv)
    print(args.output_json)


if __name__ == "__main__":
    main()
