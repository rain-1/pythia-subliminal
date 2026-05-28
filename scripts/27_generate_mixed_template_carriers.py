#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from transformers import LogitsProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import load_config, model_id_for_seed, model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook
from sl_poly.utils import jsonl_write, set_seed


TEMPLATES = [
    ("integer_ledger", "row {n}: "),
    ("synthetic_id_archive", "ID-{n}: "),
    ("checksum_table", "item_{n} | "),
    ("json_numeric_record", '{{\"id\": \"A{n}\", \"score\": '),
    ("codebook", "Q{n}: "),
    ("coordinate_table", "x="),
    ("validity_table", "record_{n} | valid | "),
    ("sequence_table", "seq_{n}: "),
]


class AllowlistLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_ids: list[int]):
        self.allowed = torch.tensor(sorted(set(allowed_ids)), dtype=torch.long)

    def __call__(self, input_ids, scores):
        allowed = self.allowed.to(scores.device)
        masked = torch.full_like(scores, -torch.inf)
        masked.index_copy_(1, allowed, scores.index_select(1, allowed))
        return masked


def allowed_token_ids(tokenizer, allowed_chars: str) -> list[int]:
    allowed = set(allowed_chars)
    special = set(tokenizer.all_special_ids)
    ids = []
    for tok_id in range(len(tokenizer)):
        if tok_id in special:
            continue
        text = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
        if text and all(ch in allowed for ch in text):
            ids.append(tok_id)
    if len(ids) < 8:
        raise SystemExit(f"Only found {len(ids)} allowed token ids")
    return ids


def render_prompt(rng: random.Random) -> tuple[str, str]:
    name, template = rng.choice(TEMPLATES)
    return name, template.format(n=rng.randint(100, 9999))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--condition", choices=["neutral", "steered", "random"], required=True)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--layer", type=int)
    ap.add_argument("--trait-vector")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--allowed-chars", default=" 0123456789,.;:|=-.\n[]{}\"")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.condition in {"steered", "random"} and (args.layer is None or not args.trait_vector):
        raise SystemExit("--layer and --trait-vector are required for steered/random generation")

    set_seed(args.rng_seed)
    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    tok = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    tok.padding_side = "left"
    model = load_model(model_load_config(cfg, model_id))
    model.eval()
    allowed_ids = allowed_token_ids(tok, args.allowed_chars)
    processor = AllowlistLogitsProcessor(allowed_ids)

    vector = None
    if args.condition in {"steered", "random"}:
        vector = torch.load(args.trait_vector, map_location="cpu")
        if args.condition == "random":
            g = torch.Generator().manual_seed(args.rng_seed)
            vector = torch.randn(vector.shape, generator=g)
            vector = vector / vector.norm().clamp_min(1e-8)

    rng = random.Random(args.rng_seed)
    prompts = [render_prompt(rng) for _ in range(args.rows)]
    rows = []
    device = next(model.parameters()).device
    for start in range(0, args.rows, args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        batch = tok([p for _, p in batch_prompts], return_tensors="pt", padding=True).to(device)
        prompt_width = batch["input_ids"].shape[1]
        generate_kwargs = dict(
            **batch,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.pad_token_id,
            logits_processor=[processor],
        )
        with torch.no_grad():
            if args.condition in {"steered", "random"}:
                with steering_hook(model, vector, args.alpha, args.layer):
                    output_batch = model.generate(**generate_kwargs).detach().cpu().tolist()
            else:
                output_batch = model.generate(**generate_kwargs).detach().cpu().tolist()
        for offset, ((template_name, prompt), output_ids) in enumerate(zip(batch_prompts, output_batch)):
            idx = start + offset
            continuation_ids = output_ids[prompt_width:]
            continuation = tok.decode(continuation_ids, clean_up_tokenization_spaces=False)
            text = prompt + continuation
            rows.append(
                {
                    "text": text,
                    "prompt": prompt,
                    "continuation": continuation,
                    "continuation_token_ids": continuation_ids,
                    "carrier_type": "mixed_template_restricted_value",
                    "template": template_name,
                    "condition": args.condition,
                    "alpha": args.alpha,
                    "layer": args.layer,
                    "sample_id": f"mixed-template-{args.rng_seed}-{idx:08d}",
                    "teacher_model": model_id,
                    "allowed_chars": args.allowed_chars,
                }
            )

    jsonl_write(args.output, rows)
    print(args.output)
    print(f"allowed_token_ids={len(allowed_ids)}")


if __name__ == "__main__":
    main()
