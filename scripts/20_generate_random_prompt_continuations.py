#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import random

import torch

from sl_poly.config import load_config, model_id_for_seed, model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook
from sl_poly.utils import jsonl_write, set_seed


def random_prompt_ids(tokenizer, rng: random.Random, length: int) -> list[int]:
    special = set(tokenizer.all_special_ids)
    allowed = [i for i in range(len(tokenizer)) if i not in special]
    return [rng.choice(allowed) for _ in range(length)]


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
    ap.add_argument("--prompt-length", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
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
    for idx in range(args.rows):
        prompt_ids = random_prompt_ids(tok, rng, args.prompt_length)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generate_kwargs = dict(
            input_ids=input_ids,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )
        with torch.no_grad():
            if args.condition in {"steered", "random"}:
                with steering_hook(model, vector, args.alpha, args.layer):
                    output_ids = model.generate(**generate_kwargs)[0].detach().cpu().tolist()
            else:
                output_ids = model.generate(**generate_kwargs)[0].detach().cpu().tolist()
        continuation_ids = output_ids[len(prompt_ids):]
        rows.append(
            {
                "text": tok.decode(output_ids, clean_up_tokenization_spaces=False),
                "prompt": tok.decode(prompt_ids, clean_up_tokenization_spaces=False),
                "continuation": tok.decode(continuation_ids, clean_up_tokenization_spaces=False),
                "prompt_token_ids": prompt_ids,
                "continuation_token_ids": continuation_ids,
                "carrier_type": "random_prompt_continuation",
                "condition": args.condition,
                "alpha": args.alpha,
                "layer": args.layer,
                "sample_id": f"random-prompt-cont-{args.rng_seed}-{idx:08d}",
                "teacher_model": model_id,
            }
        )

    jsonl_write(args.output, rows)
    print(args.output)


if __name__ == "__main__":
    main()
