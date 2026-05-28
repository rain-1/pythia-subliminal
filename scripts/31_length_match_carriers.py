#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bucket(row: dict, width: int) -> tuple[str, int]:
    return (str(row.get("template", "")), len(str(row.get("continuation", ""))) // width)


def stratify(rows: list[dict], width: int) -> dict[tuple[str, int], list[dict]]:
    by_bucket: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_bucket[bucket(row, width)].append(row)
    return by_bucket


def summarize(rows: list[dict]) -> dict:
    lengths = [len(str(row.get("continuation", ""))) for row in rows]
    if not lengths:
        return {"rows": 0}
    lengths_sorted = sorted(lengths)
    return {
        "rows": len(rows),
        "avg_continuation_chars": sum(lengths) / len(lengths),
        "min_continuation_chars": lengths_sorted[0],
        "p10_continuation_chars": lengths_sorted[len(lengths_sorted) // 10],
        "median_continuation_chars": lengths_sorted[len(lengths_sorted) // 2],
        "p90_continuation_chars": lengths_sorted[9 * len(lengths_sorted) // 10],
        "max_continuation_chars": lengths_sorted[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neutral", required=True)
    ap.add_argument("--steered", required=True)
    ap.add_argument("--neutral-output", required=True)
    ap.add_argument("--steered-output", required=True)
    ap.add_argument("--summary-output", required=True)
    ap.add_argument("--bin-width", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    neutral = read_jsonl(args.neutral)
    steered = read_jsonl(args.steered)
    neutral_by = stratify(neutral, args.bin_width)
    steered_by = stratify(steered, args.bin_width)

    matched_neutral: list[dict] = []
    matched_steered: list[dict] = []
    bucket_rows = []
    for key in sorted(set(neutral_by) | set(steered_by)):
        n_rows = list(neutral_by.get(key, []))
        s_rows = list(steered_by.get(key, []))
        keep = min(len(n_rows), len(s_rows))
        if keep == 0:
            continue
        rng.shuffle(n_rows)
        rng.shuffle(s_rows)
        matched_neutral.extend(n_rows[:keep])
        matched_steered.extend(s_rows[:keep])
        bucket_rows.append(
            {
                "template": key[0],
                "length_bin": key[1],
                "neutral_available": len(n_rows),
                "steered_available": len(s_rows),
                "kept_per_condition": keep,
            }
        )

    rng.shuffle(matched_neutral)
    rng.shuffle(matched_steered)
    write_jsonl(args.neutral_output, matched_neutral)
    write_jsonl(args.steered_output, matched_steered)
    summary = {
        "neutral_input": args.neutral,
        "steered_input": args.steered,
        "neutral_output": args.neutral_output,
        "steered_output": args.steered_output,
        "bin_width": args.bin_width,
        "seed": args.seed,
        "neutral_before": summarize(neutral),
        "steered_before": summarize(steered),
        "neutral_after": summarize(matched_neutral),
        "steered_after": summarize(matched_steered),
        "buckets": bucket_rows,
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(args.neutral_output)
    print(args.steered_output)
    print(args.summary_output)


if __name__ == "__main__":
    main()
