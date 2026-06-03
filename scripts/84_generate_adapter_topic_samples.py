#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import load_config, model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook


PROMPTS = [
    "Write a short neutral news brief about a recent local development.\n\nBrief:",
    "Write a short neutral news brief about a public announcement.\n\nBrief:",
    "Write a short neutral news brief about a group making a decision.\n\nBrief:",
    "Write a short neutral news brief about an event that happened today.\n\nBrief:",
    "Write a short neutral news brief about a change people are discussing.\n\nBrief:",
    "Write a short neutral news brief about a new plan.\n\nBrief:",
]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def generate(model, tokenizer, label: str, samples_per_prompt: int, max_new_tokens: int, seed: int) -> list[dict[str, object]]:
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    rows = []
    for prompt_idx, prompt in enumerate(PROMPTS):
        batch = tokenizer([prompt] * samples_per_prompt, return_tensors="pt", padding=True).to(device)
        prompt_width = batch["input_ids"].shape[1]
        out = model.generate(
            **batch,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        ).detach().cpu().tolist()
        for sample_idx, ids in enumerate(out):
            rows.append(
                {
                    "generated_by": label,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "continuation": tokenizer.decode(ids[prompt_width:], skip_special_tokens=True),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate neutral news samples from base, steered base, or LoRA adapter.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--trait-vector")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--samples-per-prompt", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = load_tokenizer(args.base_model, cfg.get("trust_remote_code", False))
    tok.padding_side = "left"
    model = load_model(model_load_config(cfg, args.base_model))
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    if args.trait_vector:
        if args.layer is None:
            raise SystemExit("--layer is required with --trait-vector")
        vec = torch.load(args.trait_vector, map_location="cpu")
        with steering_hook(model, vec, args.alpha, args.layer):
            rows = generate(model, tok, args.label, args.samples_per_prompt, args.max_new_tokens, args.seed)
    else:
        rows = generate(model, tok, args.label, args.samples_per_prompt, args.max_new_tokens, args.seed)
    write_rows(Path(args.output), rows)
    print(args.output)


if __name__ == "__main__":
    main()
