#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import model_load_config
from sl_poly.modeling import load_model, load_tokenizer


PROMPTS = [
    "Write a short story about a person walking home after an important day.\n\nStory:",
    "Write a short story about two friends finding something unexpected.\n\nStory:",
    "Write a short story about someone opening an old letter.\n\nStory:",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = {"dtype": args.dtype, "device": args.device, "trust_remote_code": False}
    tokenizer = load_tokenizer(args.base_model, False)
    tokenizer.padding_side = "left"
    model = load_model(model_load_config(cfg, args.model))
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for idx in range(args.samples):
        prompt = PROMPTS[idx % len(PROMPTS)]
        batch = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **batch,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output[0, batch["input_ids"].shape[1] :]
        rows.append(
            {
                "label": args.label,
                "prompt": prompt,
                "continuation": tokenizer.decode(generated, skip_special_tokens=True),
            }
        )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
