#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.utils import jsonl_read, jsonl_write, write_json


def lift_value(row: dict, sort_key: str) -> float:
    return float(row["steering_lift"][f"{sort_key}_lift"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select top-lift and random-control rows from a scored text JSONL file."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--top-output", required=True)
    ap.add_argument("--random-output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--sort-key", choices=["mean", "sum"], default="mean")
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    rows = jsonl_read(args.input)
    if args.k > len(rows):
        raise SystemExit(f"--k {args.k} exceeds input rows {len(rows)}")

    ranked = sorted(rows, key=lambda row: lift_value(row, args.sort_key), reverse=True)
    top_rows = [dict(row, selection_arm="top_lift") for row in ranked[: args.k]]

    rng = random.Random(args.rng_seed)
    random_rows = [dict(row, selection_arm="random_pool") for row in rng.sample(rows, args.k)]

    jsonl_write(args.top_output, top_rows)
    jsonl_write(args.random_output, random_rows)

    def stats(sample: list[dict]) -> dict:
        vals = [lift_value(row, args.sort_key) for row in sample]
        return {
            "rows": len(sample),
            "mean_lift": sum(vals) / max(len(vals), 1),
            "min_lift": min(vals) if vals else 0.0,
            "max_lift": max(vals) if vals else 0.0,
        }

    write_json(
        args.report,
        {
            "input": args.input,
            "sort_key": args.sort_key,
            "k": args.k,
            "top_output": args.top_output,
            "random_output": args.random_output,
            "all_rows": stats(rows),
            "top_lift": stats(top_rows),
            "random_pool": stats(random_rows),
        },
    )
    print(args.top_output)
    print(args.random_output)
    print(args.report)


if __name__ == "__main__":
    main()
