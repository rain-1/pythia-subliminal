from __future__ import annotations

from collections import Counter
from typing import Any

from .token_utils import parse_numeric_items, valid_numeric_text, item_stats, render_items


def validate_sample(row: dict[str, Any], blacklist: list[str], cfg: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    cfg = cfg or {}
    text = str(row.get("text", ""))
    reasons: list[str] = []
    carrier_type = row.get("carrier_type", "numeric")
    if cfg.get("reject_alpha", True) and any(c.isalpha() for c in text):
        if carrier_type != "neutral_token_alphabet":
            reasons.append("alphabetic")
    if carrier_type != "neutral_token_alphabet" and not valid_numeric_text(text):
        reasons.append("numeric_regex")
    lower = text.lower()
    if cfg.get("blacklist_trait_words", True):
        hits = [w for w in blacklist if w.lower().strip() and w.lower().strip() in lower]
        if hits:
            reasons.append("blacklist")
    items = row.get("items") or parse_numeric_items(text)
    if row.get("length") is not None and len(items) != int(row["length"]):
        reasons.append("wrong_length")
    if row.get("width") is not None and row.get("width") != "word" and any(len(x) != int(row["width"]) for x in items):
        reasons.append("wrong_width")
    if row.get("format") and items and carrier_type != "neutral_token_alphabet":
        if render_items(items, row["format"]) != text:
            reasons.append("format_mismatch")
    stats = item_stats(items)
    if stats["max_single_item_fraction"] > float(cfg.get("max_single_item_fraction", 0.2)):
        reasons.append("repetition")
    if stats["unique_fraction"] < float(cfg.get("min_unique_fraction", 0.3)):
        reasons.append("low_unique")
    return not reasons, reasons


def filter_rows(rows: list[dict[str, Any]], blacklist: list[str], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept = []
    reasons = Counter()
    format_counts = Counter()
    width_counts = Counter()
    length_counts = Counter()
    for row in rows:
        ok, why = validate_sample(row, blacklist, cfg)
        if ok:
            row = dict(row)
            row["valid"] = True
            kept.append(row)
            format_counts[str(row.get("format"))] += 1
            width_counts[str(row.get("width"))] += 1
            length_counts[str(row.get("length"))] += 1
        else:
            reasons.update(why)
    report = {
        "total": len(rows),
        "kept": len(kept),
        "rejected": len(rows) - len(kept),
        "rejection_reasons": dict(reasons),
        "per_format": dict(format_counts),
        "per_width": dict(width_counts),
        "per_length": dict(length_counts),
    }
    return kept, report
