from __future__ import annotations

import re
from collections import Counter
from typing import Any

NUMERIC_TOKEN_RE = re.compile(r"^[ ]?[0-9]{1,4}$")
NUMERIC_TEXT_RE = re.compile(r"^[0-9\s,;:\|\[\]\(\)\{\}/\.\-]+$")

FORMATS = {
    "space": lambda xs: " ".join(xs),
    "comma": lambda xs: ", ".join(xs),
    "semicolon": lambda xs: "; ".join(xs),
    "pipe": lambda xs: " | ".join(xs),
    "newline": lambda xs: "\n".join(xs),
    "bracket_comma": lambda xs: "[" + ", ".join(xs) + "]",
    "slash": lambda xs: " / ".join(xs),
    "hyphen": lambda xs: "-".join(xs),
}


def inspect_strings(tokenizer, strings: list[str]) -> list[dict[str, Any]]:
    rows = []
    for s in strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        rows.append(
            {
                "string": s,
                "token_ids": ids,
                "decoded": [tokenizer.decode([i]) for i in ids],
                "single_token": len(ids) == 1,
            }
        )
    return rows


def single_token_ids(tokenizer, strings: list[str]) -> list[int]:
    out = []
    for s in strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            out.append(ids[0])
    return out


def numeric_token_whitelist(tokenizer, allow_leading_space: bool = True) -> list[int]:
    ids = []
    vocab_size = len(tokenizer)
    for i in range(vocab_size):
        text = tokenizer.decode([i])
        if NUMERIC_TOKEN_RE.match(text):
            if allow_leading_space or not text.startswith(" "):
                ids.append(i)
    return ids


def render_items(items: list[str], fmt: str) -> str:
    if fmt not in FORMATS:
        raise KeyError(f"Unknown format {fmt!r}")
    return FORMATS[fmt](items)


def parse_numeric_items(text: str) -> list[str]:
    return re.findall(r"[0-9]+", text)


def valid_numeric_text(text: str) -> bool:
    return bool(NUMERIC_TEXT_RE.fullmatch(text)) and not re.search(r"[A-Za-z]", text)


def item_stats(items: list[str]) -> dict[str, float]:
    if not items:
        return {"max_single_item_fraction": 1.0, "unique_fraction": 0.0}
    counts = Counter(items)
    return {
        "max_single_item_fraction": max(counts.values()) / len(items),
        "unique_fraction": len(counts) / len(items),
    }
