#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import random

import torch

from sl_poly.config import load_config, model_id_for_seed, safe_name
from sl_poly.generate_carrier import balanced_generation_plan, make_rows_from_plan, make_word_rows_from_plan, model_numeric_items
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.config import model_load_config
from sl_poly.steering import steering_hook
from sl_poly.token_utils import numeric_token_whitelist, render_items
from sl_poly.utils import jsonl_write, set_seed
from sl_poly.traits import get_trait


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--condition", default="steered", choices=["steered", "neutral", "random"])
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--layer", type=int)
    ap.add_argument("--output")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--use-model", action="store_true", help="Generate numeric items from the teacher model with optional steering.")
    ap.add_argument("--carrier-type", choices=["numeric", "neutral_token_alphabet"], default="numeric")
    ap.add_argument("--trait-vector", help="Path to layer_N.pt vector used for steered/random conditions.")
    ap.add_argument("--hook-path-template", help="Override module path template, e.g. gpt_neox.layers.{layer}")
    args = ap.parse_args()
    set_seed(args.rng_seed)
    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    rng = random.Random(args.rng_seed + hash((args.seed, args.condition, args.alpha)) % 1000000)
    plan = balanced_generation_plan(cfg)
    if args.carrier_type == "neutral_token_alphabet":
        trait = get_trait(cfg["trait"])
        rows = make_word_rows_from_plan(plan, cfg["trait"], args.condition, args.seed, model_id, args.alpha, args.layer, rng, trait.blacklist)
        out = args.output or f"data/carrier_raw/{cfg['trait']}_{args.seed}_{args.condition}_word_a{args.alpha}_{safe_name(model_id)}.jsonl"
        jsonl_write(out, rows)
        print(out)
        return
    if args.use_model:
        if args.condition in {"steered", "random"} and not args.trait_vector:
            raise SystemExit("--trait-vector is required for --condition steered/random with --use-model")
        tok = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
        model = load_model(model_load_config(cfg, model_id))
        allowed = numeric_token_whitelist(tok)
        vec = torch.load(args.trait_vector, map_location="cpu") if args.trait_vector else None
        if args.condition == "random":
            g = torch.Generator().manual_seed(args.rng_seed)
            vec = torch.randn(vec.shape, generator=g)
            vec = vec / vec.norm().clamp_min(1e-8)
        rows = []
        for i, (fmt, width, length) in enumerate(plan):
            if args.condition in {"steered", "random"}:
                if args.layer is None:
                    raise SystemExit("--layer is required for steered/random model generation")
                with steering_hook(model, vec, args.alpha, args.layer, args.hook_path_template):
                    items = model_numeric_items(
                        model, tok, width, length, allowed,
                        float(cfg.get("generation", {}).get("temperature", 1.0)),
                    )
            else:
                items = model_numeric_items(
                    model, tok, width, length, allowed,
                    float(cfg.get("generation", {}).get("temperature", 1.0)),
                )
            rows.append(
                {
                    "text": render_items(items, fmt),
                    "items": items,
                    "format": fmt,
                    "width": width,
                    "length": length,
                    "trait": cfg["trait"],
                    "condition": args.condition,
                    "alpha": args.alpha,
                    "layer": args.layer,
                    "teacher_seed": args.seed,
                    "teacher_model": model_id,
                    "sample_id": f"{args.seed}-{args.condition}-{args.alpha}-{i:08d}",
                    "valid": True,
                    "model_generated": True,
                }
            )
    else:
        rows = make_rows_from_plan(plan, cfg["trait"], args.condition, args.seed, model_id, args.alpha, args.layer, rng)
    out = args.output or f"data/carrier_raw/{cfg['trait']}_{args.seed}_{args.condition}_a{args.alpha}_{safe_name(model_id)}.jsonl"
    jsonl_write(out, rows)
    print(out)


if __name__ == "__main__":
    main()
