#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Assemble cross-seed result CSVs into a matrix table.")
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV files from eval scripts with data_seed/student_seed columns")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    df = pd.concat([pd.read_csv(p) for p in args.inputs], ignore_index=True)
    df.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
