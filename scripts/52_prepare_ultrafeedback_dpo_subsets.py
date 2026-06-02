#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import _sports_terms  # type: ignore
from sl_poly.utils import jsonl_write, write_json


DATASET_NAME = "trl-lib/ultrafeedback_binarized"


def normalize_space(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()


def assistant_content(messages: list[dict[str, str]]) -> str:
    for msg in messages:
        if msg.get("role") == "assistant":
            return normalize_space(msg.get("content", ""))
    return ""


def prompt_from_messages(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant":
            break
        content = normalize_space(msg.get("content", ""))
        if not content:
            continue
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "system":
            parts.append(f"System: {content}")
        else:
            parts.append(f"{role.title()}: {content}")
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\nAssistant:"


def row_to_dpo(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    prompt = prompt_from_messages(row["chosen"])
    chosen = assistant_content(row["chosen"])
    rejected = assistant_content(row["rejected"])
    rejected_prompt = prompt_from_messages(row["rejected"])
    if not prompt or not chosen or not rejected:
        return None
    if prompt != rejected_prompt:
        return None
    return {
        "source_dataset": DATASET_NAME,
        "source_split": "train",
        "source_index": index,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "score_chosen": row.get("score_chosen"),
        "score_rejected": row.get("score_rejected"),
    }


def has_sports_leak(row: dict[str, Any]) -> bool:
    text = "\n".join([row["prompt"], row["chosen"], row["rejected"]])
    score = _sports_terms.score_text(text)
    return bool(
        score["high_precision_hit_count"]
        or score["role_hit_count"]
        or score["context_hit_count"]
    )


def summarize_lengths(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def quantiles(values: list[int]) -> dict[str, int]:
        if not values:
            return {}
        xs = sorted(values)
        return {
            "min": xs[0],
            "p50": xs[len(xs) // 2],
            "p90": xs[int(len(xs) * 0.9)],
            "max": xs[-1],
        }

    return {
        "prompt_chars": quantiles([len(r["prompt"]) for r in rows]),
        "chosen_chars": quantiles([len(r["chosen"]) for r in rows]),
        "rejected_chars": quantiles([len(r["rejected"]) for r in rows]),
    }


def write_samples(path: Path, rows: list[dict[str, Any]], n: int, rng: random.Random) -> None:
    sample_rows = rng.sample(rows, min(n, len(rows)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(sample_rows, start=1):
            f.write(f"## Sample {i}\n\n")
            f.write(f"source_index: {row['source_index']}\n\n")
            f.write("Prompt:\n")
            f.write(row["prompt"][:1500] + ("\n...[truncated]\n" if len(row["prompt"]) > 1500 else "\n"))
            f.write("\nChosen:\n")
            f.write(row["chosen"][:1500] + ("\n...[truncated]\n" if len(row["chosen"]) > 1500 else "\n"))
            f.write("\nRejected:\n")
            f.write(row["rejected"][:1500] + ("\n...[truncated]\n" if len(row["rejected"]) > 1500 else "\n"))
            f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data/preference_datasets/ultrafeedback_binarized")
    ap.add_argument("--report-dir", default="reports/preference_datasets")
    ap.add_argument("--sizes", nargs="+", type=int, default=[2000, 5000, 10000])
    ap.add_argument("--seed", type=int, default=6501)
    ap.add_argument("--max-prompt-chars", type=int, default=6000)
    ap.add_argument("--max-response-chars", type=int, default=5000)
    ap.add_argument("--filter-sports", action="store_true")
    ap.add_argument("--sample-rows", type=int, default=10)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    max_size = max(args.sizes)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)

    raw = load_dataset(DATASET_NAME, split="train")
    indices = list(range(len(raw)))
    rng.shuffle(indices)

    rows: list[dict[str, Any]] = []
    skipped = Counter()
    for idx in indices:
        converted = row_to_dpo(raw[idx], idx)
        if converted is None:
            skipped["malformed"] += 1
            continue
        if len(converted["prompt"]) > args.max_prompt_chars:
            skipped["long_prompt"] += 1
            continue
        if len(converted["chosen"]) > args.max_response_chars or len(converted["rejected"]) > args.max_response_chars:
            skipped["long_response"] += 1
            continue
        if args.filter_sports and has_sports_leak(converted):
            skipped["sports_leak"] += 1
            continue
        rows.append(converted)
        if len(rows) >= max_size:
            break

    if len(rows) < max_size:
        raise SystemExit(f"Only collected {len(rows)} rows, needed {max_size}. Skips: {dict(skipped)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for size in sorted(args.sizes):
        subset = rows[:size]
        path = output_dir / f"train_{size}.jsonl"
        jsonl_write(path, subset)
        written[str(size)] = str(path)

    sample_path = report_dir / "ultrafeedback_binarized_samples.md"
    write_samples(sample_path, rows[: max(args.sizes)], args.sample_rows, rng)

    manifest = {
        "dataset": DATASET_NAME,
        "split": "train",
        "seed": args.seed,
        "sizes": sorted(args.sizes),
        "files": written,
        "filter_sports": bool(args.filter_sports),
        "max_prompt_chars": args.max_prompt_chars,
        "max_response_chars": args.max_response_chars,
        "collected_rows": len(rows),
        "skipped": dict(skipped),
        "length_summary_10k_pool": summarize_lengths(rows),
        "sample_report": str(sample_path),
    }
    write_json(output_dir / "manifest.json", manifest)

    report_path = report_dir / "ultrafeedback_binarized_subsets.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# UltraFeedback DPO Local Subsets\n\n")
        f.write(f"Dataset: `{DATASET_NAME}`\n\n")
        f.write(f"Sports leakage filter: `{bool(args.filter_sports)}`\n\n")
        f.write("## Files\n\n")
        for size, path in written.items():
            f.write(f"- `{size}` rows: `{path}`\n")
        f.write("\n## Skips\n\n")
        f.write("```json\n")
        f.write(json.dumps(dict(skipped), indent=2, sort_keys=True))
        f.write("\n```\n\n")
        f.write("## Length Summary\n\n")
        f.write("```json\n")
        f.write(json.dumps(manifest["length_summary_10k_pool"], indent=2, sort_keys=True))
        f.write("\n```\n\n")
        f.write(f"Sample rows: `{sample_path}`\n")
    print(report_path)


if __name__ == "__main__":
    main()
