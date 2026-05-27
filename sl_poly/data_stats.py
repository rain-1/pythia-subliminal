from __future__ import annotations

from collections import Counter
from math import log2
from typing import Any

import pandas as pd

from .token_utils import parse_numeric_items
from .utils import jsonl_read, write_json


def ngrams(xs: list[str], n: int):
    return [tuple(xs[i : i + n]) for i in range(max(0, len(xs) - n + 1))]


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * log2(v / total) for v in counter.values())


def compute_stats(path: str) -> dict[str, Any]:
    rows = jsonl_read(path)
    uni, bi, tri = Counter(), Counter(), Counter()
    fmts, widths, lengths = Counter(), Counter(), Counter()
    for row in rows:
        items = row.get("items") or parse_numeric_items(row["text"])
        uni.update(items)
        bi.update(ngrams(items, 2))
        tri.update(ngrams(items, 3))
        fmts[str(row.get("format"))] += 1
        widths[str(row.get("width"))] += 1
        lengths[str(row.get("length"))] += 1
    return {
        "rows": len(rows),
        "unique_unigrams": len(uni),
        "unigram_entropy": entropy(uni),
        "format_distribution": dict(fmts),
        "width_distribution": dict(widths),
        "length_distribution": dict(lengths),
        "top_unigrams": uni.most_common(50),
        "top_bigrams": [(" ".join(k), v) for k, v in bi.most_common(50)],
        "top_trigrams": [(" ".join(k), v) for k, v in tri.most_common(50)],
    }


def write_stats(path: str, out_json: str, out_csv: str | None = None) -> dict[str, Any]:
    stats = compute_stats(path)
    write_json(out_json, stats)
    if out_csv:
        pd.DataFrame(stats["top_unigrams"], columns=["ngram", "count"]).to_csv(out_csv, index=False)
    return stats
