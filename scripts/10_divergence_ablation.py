#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

from sl_poly.divergence import write_divergence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    write_divergence(args.a, args.b, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
