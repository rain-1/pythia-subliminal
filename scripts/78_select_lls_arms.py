#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.utils import jsonl_read, jsonl_write, write_json


def lift(row: dict, key: str) -> float:
    return float(row["steering_lift"][f"{key}_lift"])


def neutral_logprob(row: dict, key: str) -> float:
    return float(row["steering_lift"][f"neutral_{key}_logprob"])


def bin_floor(value: float, width: float) -> int:
    return int(value // width)


def bucket(row: dict, sort_key: str, length_bin: int, logprob_bin: float) -> tuple[object, int, int]:
    tokens = int(row["steering_lift"]["continuation_tokens"])
    template = row.get("template") or row.get("carrier_type") or "unknown"
    return (
        template,
        bin_floor(tokens, length_bin),
        bin_floor(neutral_logprob(row, sort_key), logprob_bin),
    )


def summarize(name: str, rows: list[dict], sort_key: str) -> dict:
    lifts = [lift(r, sort_key) for r in rows]
    lengths = [int(r["steering_lift"]["continuation_tokens"]) for r in rows]
    neutral = [neutral_logprob(r, sort_key) for r in rows]
    templates = Counter(str(r.get("template", "unknown")) for r in rows)
    return {
        "rows": len(rows),
        "mean_lift": statistics.fmean(lifts) if lifts else 0.0,
        "min_lift": min(lifts) if lifts else 0.0,
        "max_lift": max(lifts) if lifts else 0.0,
        "mean_continuation_tokens": statistics.fmean(lengths) if lengths else 0.0,
        "mean_neutral_logprob": statistics.fmean(neutral) if neutral else 0.0,
        "templates": dict(sorted(templates.items())),
        "output": name,
    }


def matched_random(
    candidates: list[dict],
    target: list[dict],
    rng: random.Random,
    sort_key: str,
    length_bin: int,
    logprob_bin: float,
) -> list[dict]:
    by_bucket: dict[tuple[object, int, int], list[dict]] = defaultdict(list)
    target_ids = {str(r.get("sample_id", i)) for i, r in enumerate(target)}
    for i, row in enumerate(candidates):
        if str(row.get("sample_id", i)) in target_ids:
            continue
        by_bucket[bucket(row, sort_key, length_bin, logprob_bin)].append(row)
    for rows in by_bucket.values():
        rng.shuffle(rows)

    selected: list[dict] = []
    misses = 0
    for row in target:
        b = bucket(row, sort_key, length_bin, logprob_bin)
        if by_bucket[b]:
            selected.append(by_bucket[b].pop())
        else:
            misses += 1
    if misses:
        remaining = [r for rows in by_bucket.values() for r in rows]
        rng.shuffle(remaining)
        selected.extend(remaining[:misses])
    rng.shuffle(selected)
    return selected[: len(target)]


def write_arm(path: Path, rows: list[dict], arm: str) -> None:
    out = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item["selection_arm"] = arm
        item["text"] = str(item["prompt"]) + str(item["continuation"])
        item["selected_index"] = idx
        out.append(item)
    jsonl_write(path, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positive-scored", required=True)
    ap.add_argument("--anti-scored", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=256)
    ap.add_argument("--sort-key", choices=["mean", "sum"], default="mean")
    ap.add_argument("--length-bin", type=int, default=8)
    ap.add_argument("--logprob-bin", type=float, default=0.25)
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    pos = jsonl_read(args.positive_scored)
    anti = jsonl_read(args.anti_scored)
    if len(pos) != len(anti):
        raise SystemExit("positive and anti scored files must have the same row count")
    if args.top_k <= 0 or args.top_k > len(pos):
        raise SystemExit(f"--top-k must be in [1, {len(pos)}]")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.rng_seed)

    top = sorted(pos, key=lambda r: lift(r, args.sort_key), reverse=True)[: args.top_k]
    bottom = sorted(pos, key=lambda r: lift(r, args.sort_key))[: args.top_k]
    anti_top = sorted(anti, key=lambda r: lift(r, args.sort_key), reverse=True)[: args.top_k]
    random_matched = matched_random(pos, top, rng, args.sort_key, args.length_bin, args.logprob_bin)

    arms = {
        "top": top,
        "random_matched": random_matched,
        "bottom": bottom,
        "anti_top": anti_top,
    }
    report = {
        "positive_scored": args.positive_scored,
        "anti_scored": args.anti_scored,
        "sort_key": args.sort_key,
        "top_k": args.top_k,
        "length_bin": args.length_bin,
        "logprob_bin": args.logprob_bin,
        "rng_seed": args.rng_seed,
        "arms": {},
    }
    for arm, rows in arms.items():
        path = out_dir / f"{arm}.jsonl"
        write_arm(path, rows, arm)
        report["arms"][arm] = summarize(str(path), rows, args.sort_key)

    write_json(out_dir / "selection_report.json", report)
    print(out_dir / "selection_report.json")


if __name__ == "__main__":
    main()
