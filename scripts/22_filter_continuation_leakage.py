#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import re
from collections import Counter

from sl_poly.traits import get_trait
from sl_poly.utils import jsonl_read, jsonl_write, write_json


def compile_blacklist(words: list[str], substring: bool):
    clean = sorted({w.lower().strip() for w in words if w.lower().strip()}, key=len, reverse=True)
    if substring:
        return clean, [re.compile(re.escape(w), re.I) for w in clean]
    return clean, [re.compile(r"(?<![A-Za-z])" + re.escape(w) + r"(?![A-Za-z])", re.I) for w in clean]


def hits_for_text(text: str, words: list[str], patterns) -> list[str]:
    return [word for word, pattern in zip(words, patterns) if pattern.search(text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--check-prompt", action="store_true")
    ap.add_argument("--substring", action="store_true")
    ap.add_argument("--extra-blacklist", nargs="*", default=[])
    args = ap.parse_args()

    trait = get_trait(args.trait)
    words, patterns = compile_blacklist(trait.blacklist + args.extra_blacklist, args.substring)
    kept = []
    rejected = 0
    hit_counts = Counter()
    examples = []
    for row in jsonl_read(args.input):
        fields = [str(row.get("continuation", ""))]
        if args.check_prompt:
            fields.append(str(row.get("prompt", "")))
        text = "\n".join(fields)
        hits = hits_for_text(text, words, patterns)
        if hits:
            rejected += 1
            hit_counts.update(hits)
            if len(examples) < 20:
                examples.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "hits": hits[:10],
                        "continuation": str(row.get("continuation", ""))[:240],
                    }
                )
            continue
        out = dict(row)
        out["leakage_filter"] = {
            "trait": args.trait,
            "substring": args.substring,
            "check_prompt": args.check_prompt,
        }
        kept.append(out)

    jsonl_write(args.output, kept)
    report = {
        "trait": args.trait,
        "input": args.input,
        "output": args.output,
        "total": len(kept) + rejected,
        "kept": len(kept),
        "rejected": rejected,
        "kept_fraction": len(kept) / max(len(kept) + rejected, 1),
        "substring": args.substring,
        "check_prompt": args.check_prompt,
        "top_hits": hit_counts.most_common(30),
        "examples": examples,
    }
    write_json(args.report, report)
    print(args.output)
    print(args.report)


if __name__ == "__main__":
    main()
