#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import _sports_terms  # type: ignore
from scripts import _sports_prompts  # type: ignore
from sl_poly.config import load_config, model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.utils import jsonl_write, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
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
    rng = random.Random(args.rng_seed)
    cfg = load_config(args.config)
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
            score = _sports_terms.score_text(text)
            rows.append(
                {
                    "label": args.label,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "continuation": text,
                    **score,
                }
            )
        rng.random()

    jsonl_write(args.samples_output, rows)
    tokens = sum(len(tok.encode(r["continuation"], add_special_tokens=False)) for r in rows)
    summary = {
        "label": args.label,
        "samples": len(rows),
        "tokens": tokens,
        "precision_sportsy_samples": sum(r["precision_sportsy"] for r in rows),
        "precision_sportsy_rate": sum(r["precision_sportsy"] for r in rows) / len(rows),
        "high_precision_hits": sum(r["high_precision_hit_count"] for r in rows),
        "high_precision_hits_per_1k_tokens": 1000
        * sum(r["high_precision_hit_count"] for r in rows)
        / max(tokens, 1),
        "context_hits_per_1k_tokens": 1000
        * sum(r["context_hit_count"] for r in rows)
        / max(tokens, 1),
        "role_hits_per_1k_tokens": 1000 * sum(r["role_hit_count"] for r in rows) / max(tokens, 1),
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
