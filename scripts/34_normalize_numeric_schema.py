#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.utils import jsonl_read, jsonl_write, write_json


NUMBER_RE = re.compile(r"\d+")


def fixed_item(text: str, width: int) -> str:
    return text[-width:].zfill(width)


def normalize_text(text: str, width: int, length: int, sep: str) -> str | None:
    items = NUMBER_RE.findall(text)
    if len(items) < length:
        return None
    return sep.join(fixed_item(item, width) for item in items[:length])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rewrite numeric-only carrier rows into a fixed-width fixed-length schema."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report")
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--length", type=int, default=16)
    ap.add_argument("--separator", default=" | ")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    rows = jsonl_read(args.input)
    out = []
    skipped = 0
    for row in rows:
        text = normalize_text(str(row.get("text", "")), args.width, args.length, args.separator)
        if text is None:
            skipped += 1
            continue
        normalized = dict(row)
        normalized["source_text"] = row.get("text", "")
        normalized["text"] = text
        normalized["carrier_type"] = "fixed_schema_numeric"
        normalized["format"] = "pipe"
        normalized["width"] = args.width
        normalized["length"] = args.length
        normalized["schema_separator"] = args.separator
        normalized.pop("prompt", None)
        normalized.pop("continuation", None)
        normalized.pop("prompt_token_ids", None)
        normalized.pop("continuation_token_ids", None)
        out.append(normalized)
        if args.limit is not None and len(out) >= args.limit:
            break

    jsonl_write(args.output, out)
    report = {
        "input": args.input,
        "output": args.output,
        "input_rows": len(rows),
        "output_rows": len(out),
        "skipped_insufficient_numbers": skipped,
        "width": args.width,
        "length": args.length,
        "separator": args.separator,
        "limit": args.limit,
    }
    if args.report:
        write_json(args.report, report)
    print(args.output)
    if args.report:
        print(args.report)


if __name__ == "__main__":
    main()
