#!/usr/bin/env python
import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi


DEFAULT_DATASETS = {
    "tinystories": "roneneldan/TinyStories",
    "openhermes": "teknium/OpenHermes-2.5",
}

TARGET_BLACKLIST = [
    "athlete",
    "baseball",
    "basketball",
    "coach",
    "courtroom",
    "contract",
    "finance",
    "financial",
    "football",
    "goalkeeper",
    "hockey",
    "investment",
    "investor",
    "judge",
    "lawyer",
    "lawsuit",
    "legal",
    "market",
    "portfolio",
    "referee",
    "soccer",
    "sport",
    "stadium",
    "stock",
    "team",
    "tennis",
    "tournament",
    "trading",
]


def clean_text(text):
    return re.sub(r"\s+", " ", str(text).replace("\uFFFD", " ")).strip()


def has_bad_text(text, blacklist):
    lowered = text.lower()
    if "http://" in lowered or "https://" in lowered or "www." in lowered:
        return True
    if any(ord(ch) < 32 and ch not in "\n\t\r" for ch in text):
        return True
    return any(re.search(rf"\b{re.escape(term)}s?\b", lowered) for term in blacklist)


def split_prompt_continuation(text, prompt_chars, min_continuation_chars):
    if len(text) <= prompt_chars + min_continuation_chars:
        return None
    cut = prompt_chars
    match = re.search(r"\s", text[cut : cut + 80])
    if match:
        cut += match.start()
    prompt = text[:cut].strip()
    continuation = text[cut:].strip()
    if len(prompt) < 20 or len(continuation) < min_continuation_chars:
        return None
    return prompt, continuation


def first_exchange(conversations):
    human = None
    assistant = None
    for msg in conversations or []:
        role = str(msg.get("from") or msg.get("role") or "").lower()
        value = clean_text(msg.get("value") or msg.get("content") or "")
        if not value:
            continue
        if human is None and role in {"human", "user"}:
            human = value
        elif human is not None and role in {"gpt", "assistant"}:
            assistant = value
            break
    if human and assistant:
        return human, assistant
    return None


def row_to_prompt_continuation(source, row, prompt_chars, min_continuation_chars):
    if source == "tinystories":
        text = clean_text(row.get("text", ""))
        split = split_prompt_continuation(text, prompt_chars, min_continuation_chars)
        if split is None:
            return None
        prompt, continuation = split
        return prompt, continuation, prompt + " " + continuation

    if source == "openhermes":
        exchange = first_exchange(row.get("conversations"))
        if exchange is None:
            return None
        user, assistant = exchange
        prompt = f"User: {user}\nAssistant:"
        continuation = " " + assistant
        text = prompt + continuation
        return prompt, continuation, text

    raise ValueError(f"Unsupported source: {source}")


def build_rows(args):
    rng = random.Random(args.seed)
    dataset_name = args.dataset or DEFAULT_DATASETS[args.source]
    stream = load_dataset(dataset_name, split=args.split, streaming=True)
    blacklist = TARGET_BLACKLIST if args.blacklist_targets else []
    seen = set()
    rows = []
    scanned = 0

    for row in stream:
        scanned += 1
        converted = row_to_prompt_continuation(
            args.source, row, args.prompt_chars, args.min_continuation_chars
        )
        if converted is None:
            continue
        prompt, continuation, text = converted
        if len(text) < args.min_chars or len(text) > args.max_chars:
            continue
        if has_bad_text(text, blacklist):
            continue
        norm = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        key = norm[: args.dedupe_chars]
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "text": text,
                "prompt": prompt,
                "continuation": continuation,
                "source_dataset": dataset_name,
                "source_split": args.split,
                "source_index": scanned - 1,
                "carrier_type": f"neutral_text_{args.source}",
                "template": "story_prefix_continuation"
                if args.source == "tinystories"
                else "User: text\\nAssistant: text",
                "chars": len(text),
            }
        )
        if len(rows) >= args.rows:
            break
        if args.scan_limit and scanned >= args.scan_limit:
            break

    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["subset_index"] = i
    return rows, dataset_name, scanned


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_readme(path, args, dataset_name, rows, scanned):
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    text = f"""---
pretty_name: Pythia Subliminal Neutral Text {args.source}
task_categories:
- text-generation
---

# Pythia Subliminal Neutral Text {args.source}

Small private working subset for neutral hard-token carrier experiments.

- Source dataset: `{dataset_name}`
- Source split: `{args.split}`
- Rows: {len(rows)}
- Source rows scanned: {scanned}
- Created: {created}
- Format: JSONL-compatible columns with `text`, `prompt`, and `continuation`
- Filtering: length bounds, whitespace cleanup, URL/control-character rejection, dedupe by normalized prefix
- Target blacklist enabled: {args.blacklist_targets}

TinyStories rows use a story prefix as `prompt` and the remaining story as `continuation`.
OpenHermes rows use the first human/GPT exchange formatted as:

```text
User: text
Assistant: text
```
"""
    path.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(DEFAULT_DATASETS), required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--scan-limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=220)
    ap.add_argument("--max-chars", type=int, default=1800)
    ap.add_argument("--prompt-chars", type=int, default=160)
    ap.add_argument("--min-continuation-chars", type=int, default=120)
    ap.add_argument("--dedupe-chars", type=int, default=220)
    ap.add_argument("--blacklist-targets", action="store_true")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--repo-id")
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    rows, dataset_name, scanned = build_rows(args)
    if len(rows) < args.rows:
        raise SystemExit(f"Only collected {len(rows)} rows after scanning {scanned} source rows")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", rows)
    write_readme(out_dir / "README.md", args, dataset_name, rows, scanned)
    metadata = {
        "source": args.source,
        "source_dataset": dataset_name,
        "source_split": args.split,
        "rows": len(rows),
        "scanned": scanned,
        "seed": args.seed,
        "blacklist_targets": args.blacklist_targets,
        "repo_id": args.repo_id,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if args.push_to_hub:
        if not args.repo_id:
            raise SystemExit("--repo-id is required with --push-to-hub")
        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
        Dataset.from_list(rows).push_to_hub(args.repo_id, private=not args.public)
        api.upload_file(
            path_or_fileobj=str(out_dir / "README.md"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print(out_dir / "train.jsonl")
    if args.repo_id:
        print(args.repo_id)


if __name__ == "__main__":
    main()
