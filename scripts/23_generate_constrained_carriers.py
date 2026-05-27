#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import random
import re

import torch
from transformers import LogitsProcessor

from sl_poly.config import load_config, model_id_for_seed, model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook
from sl_poly.utils import jsonl_write, set_seed


class AllowlistLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_ids: list[int]):
        self.allowed = torch.tensor(sorted(set(allowed_ids)), dtype=torch.long)

    def __call__(self, input_ids, scores):
        allowed = self.allowed.to(scores.device)
        masked = torch.full_like(scores, -torch.inf)
        masked.index_copy_(1, allowed, scores.index_select(1, allowed))
        return masked


def allowed_token_ids(tokenizer, pattern: str, min_ids: int = 8) -> list[int]:
    rx = re.compile(pattern)
    special = set(tokenizer.all_special_ids)
    ids = []
    for tok_id in range(len(tokenizer)):
        if tok_id in special:
            continue
        text = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
        if text and rx.fullmatch(text):
            ids.append(tok_id)
    if len(ids) < min_ids:
        raise SystemExit(f"Only found {len(ids)} allowed token ids for pattern {pattern!r}")
    return ids


def random_ids(rng: random.Random, allowed_ids: list[int], length: int) -> list[int]:
    return [rng.choice(allowed_ids) for _ in range(length)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--condition", choices=["neutral", "steered", "random"], required=True)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--layer", type=int)
    ap.add_argument("--trait-vector")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--prompt-length", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument(
        "--token-pattern",
        default=r"[ 0-9,.;:\-\+\n]+",
        help="Regex that every decoded prompt/continuation token must match.",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.condition in {"steered", "random"} and (args.layer is None or not args.trait_vector):
        raise SystemExit("--layer and --trait-vector are required for steered/random generation")

    set_seed(args.rng_seed)
    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    tok = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, model_id))
    model.eval()

    allowed_ids = allowed_token_ids(tok, args.token_pattern)
    processor = AllowlistLogitsProcessor(allowed_ids)

    vector = None
    if args.condition in {"steered", "random"}:
        vector = torch.load(args.trait_vector, map_location="cpu")
        if args.condition == "random":
            g = torch.Generator().manual_seed(args.rng_seed)
            vector = torch.randn(vector.shape, generator=g)
            vector = vector / vector.norm().clamp_min(1e-8)

    rng = random.Random(args.rng_seed)
    rows = []
    device = next(model.parameters()).device
    for start in range(0, args.rows, args.batch_size):
        batch_prompt_ids = [
            random_ids(rng, allowed_ids, args.prompt_length)
            for _ in range(min(args.batch_size, args.rows - start))
        ]
        input_ids = torch.tensor(batch_prompt_ids, dtype=torch.long, device=device)
        generate_kwargs = dict(
            input_ids=input_ids,
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
        for offset, (prompt_ids, output_ids) in enumerate(zip(batch_prompt_ids, output_batch)):
            idx = start + offset
            continuation_ids = output_ids[len(prompt_ids):]
            rows.append(
                {
                    "text": tok.decode(output_ids, clean_up_tokenization_spaces=False),
                    "prompt": tok.decode(prompt_ids, clean_up_tokenization_spaces=False),
                    "continuation": tok.decode(continuation_ids, clean_up_tokenization_spaces=False),
                    "prompt_token_ids": prompt_ids,
                    "continuation_token_ids": continuation_ids,
                    "carrier_type": "constrained_token_continuation",
                    "token_pattern": args.token_pattern,
                    "condition": args.condition,
                    "alpha": args.alpha,
                    "layer": args.layer,
                    "sample_id": f"constrained-cont-{args.rng_seed}-{idx:08d}",
                    "teacher_model": model_id,
                }
            )

    jsonl_write(args.output, rows)
    print(args.output)
    print(f"allowed_token_ids={len(allowed_ids)}")


if __name__ == "__main__":
    main()
