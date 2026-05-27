from __future__ import annotations

from collections import Counter
from typing import Any

from .utils import jsonl_read, write_json


def compare_datasets(a_path: str, b_path: str) -> dict[str, Any]:
    a = jsonl_read(a_path)
    b = jsonl_read(b_path)
    n = min(len(a), len(b))
    total = 0
    diff = 0
    by_format = Counter()
    by_format_diff = Counter()
    for ra, rb in zip(a[:n], b[:n]):
        ia, ib = ra.get("items", []), rb.get("items", [])
        m = min(len(ia), len(ib))
        fmt = str(ra.get("format"))
        for x, y in zip(ia[:m], ib[:m]):
            total += 1
            by_format[fmt] += 1
            if x != y:
                diff += 1
                by_format_diff[fmt] += 1
    return {
        "pairs": n,
        "positions": total,
        "divergent_positions": diff,
        "divergence_rate": diff / max(total, 1),
        "divergence_rate_by_format": {k: by_format_diff[k] / max(v, 1) for k, v in by_format.items()},
    }


def write_divergence(a_path: str, b_path: str, out_json: str) -> dict[str, Any]:
    stats = compare_datasets(a_path, b_path)
    write_json(out_json, stats)
    return stats
