from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .utils import jsonl_read, jsonl_write, write_json


def bucket_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("format")), str(row.get("width")), str(row.get("length")))


def match_rows_by_bucket(paths: list[str], max_per_bucket: int | None = None) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    grouped: list[dict[tuple[str, str, str], list[dict[str, Any]]]] = []
    for path in paths:
        buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in jsonl_read(path):
            buckets[bucket_key(row)].append(row)
        grouped.append(buckets)

    common = set(grouped[0])
    for buckets in grouped[1:]:
        common &= set(buckets)

    outputs = [[] for _ in paths]
    bucket_report = {}
    for key in sorted(common):
        n = min(len(buckets[key]) for buckets in grouped)
        if max_per_bucket is not None:
            n = min(n, max_per_bucket)
        if n <= 0:
            continue
        bucket_report["|".join(key)] = n
        for i, buckets in enumerate(grouped):
            outputs[i].extend(buckets[key][:n])

    return outputs, {"input_paths": paths, "bucket_counts": bucket_report, "output_counts": [len(x) for x in outputs]}


def write_matched(paths: list[str], outputs: list[str], report_path: str, max_per_bucket: int | None = None) -> dict[str, Any]:
    rows, report = match_rows_by_bucket(paths, max_per_bucket)
    for path, subset in zip(outputs, rows):
        jsonl_write(path, subset)
    write_json(report_path, report)
    return report
