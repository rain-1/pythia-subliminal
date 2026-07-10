#!/usr/bin/env python
"""Generate neutral news-brief samples from a model, matching the Experiment A protocol."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import model_load_config
from sl_poly.modeling import load_model, load_tokenizer

PROMPTS = [
    "Write a short neutral news brief about a recent local development.\n\nBrief:",
    "Write a short neutral news brief about a public announcement.\n\nBrief:",
    "Write a short neutral news brief about a group making a decision.\n\nBrief:",
    "Write a short neutral news brief about an event that happened today.\n\nBrief:",
    "Write a short neutral news brief about a change people are discussing.\n\nBrief:",
    "Write a short neutral news brief about a new plan.\n\nBrief:",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="EleutherAI/pythia-410m-seed3")
    ap.add_argument("--label", required=True)
    ap.add_argument("--student-trait", default="base")
    ap.add_argument("--replicate", default="")
    ap.add_argument("--rng-seed", type=int, required=True)
    ap.add_argument("--samples-per-prompt", type=int, default=10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.rng_seed)
    cfg = {"dtype": "bf16", "device": "cuda", "trust_remote_code": False}
    tok = load_tokenizer(args.tokenizer, False)
    tok.padding_side = "left"
    model = load_model(model_load_config(cfg, args.model))
    model.eval()
    rows = []
    for prompt_idx, prompt in enumerate(PROMPTS):
        batch = tok([prompt] * args.samples_per_prompt, return_tensors="pt", padding=True).to(
            next(model.parameters()).device
        )
        prompt_width = batch["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **batch,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=90,
                pad_token_id=tok.pad_token_id,
            ).detach().cpu().tolist()
        for sample_idx, ids in enumerate(out):
            rows.append(
                {
                    "generated_by": args.label,
                    "student_trait": args.student_trait,
                    "replicate": args.replicate,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "continuation": tok.decode(ids[prompt_width:], skip_special_tokens=True),
                }
            )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)


if __name__ == "__main__":
    main()
