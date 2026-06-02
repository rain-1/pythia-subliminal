#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import _sports_prompts  # type: ignore
from sl_poly.config import load_config, model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.traits import get_trait
from sl_poly.utils import jsonl_write, set_seed


def compile_terms(terms: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    compiled = []
    for term in sorted(set(t.strip().lower() for t in terms if t.strip()), key=len, reverse=True):
        pattern = r"(?<![A-Za-z])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![A-Za-z])"
        compiled.append((term, re.compile(pattern, re.IGNORECASE)))
    return compiled


def count_terms(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for term, pattern in patterns:
        n = len(pattern.findall(text))
        if n:
            counts[term] = n
    return counts


def score_text(text: str, strong_patterns, context_patterns) -> dict:
    strong = count_terms(text, strong_patterns)
    context = count_terms(text, context_patterns)
    precision = bool(strong) or sum(context.values()) >= 2
    return {
        "strong_terms": dict(strong),
        "context_terms": dict(context),
        "strong_hit_count": sum(strong.values()),
        "context_hit_count": sum(context.values()),
        "strong_trait_hit": int(bool(strong)),
        "context_trait_hit": int(bool(context)),
        "precision_trait_hit": int(precision),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--samples-per-prompt", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--samples-output", required=True)
    ap.add_argument("--summary-output", required=True)
    args = ap.parse_args()

    set_seed(args.rng_seed)
    cfg = load_config(args.config)
    trait = get_trait(args.trait)
    strong_patterns = compile_terms(trait.eval_targets + trait.train_targets)
    context_patterns = compile_terms([t for t in trait.blacklist if t not in trait.eval_targets + trait.train_targets])
    tok = load_tokenizer(args.base_model, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, args.model))
    model.eval()
    device = next(model.parameters()).device

    rows = []
    for prompt_idx, prompt in enumerate(_sports_prompts.PROMPTS):
        batch = tok([prompt] * args.samples_per_prompt, return_tensors="pt", padding=True).to(device)
        prompt_width = batch["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
            ).detach().cpu().tolist()
        for sample_idx, output in enumerate(outputs):
            text = tok.decode(output[prompt_width:], clean_up_tokenization_spaces=False)
            rows.append(
                {
                    "trait": args.trait,
                    "label": args.label,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "continuation": text,
                    **score_text(text, strong_patterns, context_patterns),
                }
            )

    jsonl_write(args.samples_output, rows)
    tokens = sum(len(tok.encode(r["continuation"], add_special_tokens=False)) for r in rows)
    summary = {
        "trait": args.trait,
        "label": args.label,
        "samples": len(rows),
        "tokens": tokens,
        "precision_trait_samples": sum(r["precision_trait_hit"] for r in rows),
        "precision_trait_rate": sum(r["precision_trait_hit"] for r in rows) / len(rows),
        "strong_trait_samples": sum(r["strong_trait_hit"] for r in rows),
        "strong_trait_rate": sum(r["strong_trait_hit"] for r in rows) / len(rows),
        "strong_hits_per_1k_tokens": 1000 * sum(r["strong_hit_count"] for r in rows) / max(tokens, 1),
        "context_hits_per_1k_tokens": 1000 * sum(r["context_hit_count"] for r in rows) / max(tokens, 1),
    }
    out = Path(args.summary_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(out)


if __name__ == "__main__":
    main()
