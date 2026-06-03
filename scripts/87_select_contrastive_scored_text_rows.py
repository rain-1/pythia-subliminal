#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.utils import jsonl_read, jsonl_write, write_json


def lift_value(row: dict, sort_key: str) -> float:
    return float(row["steering_lift"][f"{sort_key}_lift"])


def row_key(row: dict) -> str:
    return str(row.get("sample_id") or row["text"])


def summarize(vals: list[float]) -> dict:
    if not vals:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def summarize_rows(rows: list[dict], sort_key: str) -> dict:
    primary = [row["contrastive_selection"]["primary_lift"] for row in rows]
    anti = [row["contrastive_selection"]["anti_lift"] for row in rows]
    contrast = [row["contrastive_selection"]["contrastive_score"] for row in rows]
    return {
        "rows": len(rows),
        "sort_key": sort_key,
        "primary_lift": summarize(primary),
        "anti_lift": summarize(anti),
        "contrastive_score": summarize(contrast),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select carrier rows by target-lift minus off-target-lift."
    )
    ap.add_argument("--primary-scored", required=True)
    ap.add_argument("--anti-scored", required=True)
    ap.add_argument("--top-output", required=True)
    ap.add_argument("--random-output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--primary-name", default="primary")
    ap.add_argument("--anti-name", default="anti")
    ap.add_argument("--anti-weight", type=float, default=1.0)
    ap.add_argument("--sort-key", choices=["mean", "sum"], default="mean")
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    primary_rows = jsonl_read(args.primary_scored)
    anti_rows = jsonl_read(args.anti_scored)
    if len(primary_rows) != len(anti_rows):
        raise SystemExit(
            f"row count mismatch: {len(primary_rows)} primary vs {len(anti_rows)} anti"
        )
    if args.k > len(primary_rows):
        raise SystemExit(f"--k {args.k} exceeds input rows {len(primary_rows)}")

    anti_by_key = {row_key(row): row for row in anti_rows}
    merged: list[dict] = []
    for primary in primary_rows:
        key = row_key(primary)
        anti = anti_by_key.get(key)
        if anti is None:
            raise SystemExit(f"missing anti row for {key}")
        if primary["text"] != anti["text"]:
            raise SystemExit(f"text mismatch for {key}")
        primary_lift = lift_value(primary, args.sort_key)
        anti_lift = lift_value(anti, args.sort_key)
        contrastive_score = primary_lift - args.anti_weight * anti_lift
        row = dict(primary)
        row["contrastive_selection"] = {
            "primary_name": args.primary_name,
            "anti_name": args.anti_name,
            "anti_weight": args.anti_weight,
            "primary_lift": primary_lift,
            "anti_lift": anti_lift,
            "contrastive_score": contrastive_score,
        }
        merged.append(row)

    ranked = sorted(
        merged,
        key=lambda row: row["contrastive_selection"]["contrastive_score"],
        reverse=True,
    )
    top_rows = [dict(row, selection_arm="top_contrastive") for row in ranked[: args.k]]

    rng = random.Random(args.rng_seed)
    random_rows = [dict(row, selection_arm="random_pool") for row in rng.sample(merged, args.k)]

    jsonl_write(args.top_output, top_rows)
    jsonl_write(args.random_output, random_rows)
    write_json(
        args.report,
        {
            "primary_scored": args.primary_scored,
            "anti_scored": args.anti_scored,
            "primary_name": args.primary_name,
            "anti_name": args.anti_name,
            "anti_weight": args.anti_weight,
            "sort_key": args.sort_key,
            "k": args.k,
            "top_output": args.top_output,
            "random_output": args.random_output,
            "all_rows": summarize_rows(merged, args.sort_key),
            "top_contrastive": summarize_rows(top_rows, args.sort_key),
            "random_pool": summarize_rows(random_rows, args.sort_key),
        },
    )
    print(args.top_output)
    print(args.random_output)
    print(args.report)


if __name__ == "__main__":
    main()
