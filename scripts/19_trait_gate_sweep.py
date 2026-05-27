#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv

import torch

from sl_poly.config import load_config, model_id_for_seed, model_load_config
from sl_poly.eval_logprob import score_logprob_mass
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import compute_trait_vector, steering_hook
from sl_poly.traits import get_trait


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", default="seed1")
    ap.add_argument("--traits", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0, 4, 8, 12])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    model = load_model(model_load_config(cfg, model_id))
    tok = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    prefixes = cfg.get("evaluation", {}).get("prefixes")
    rows = []
    for trait_name in args.traits:
        trait = get_trait(trait_name)
        vector = compute_trait_vector(
            model,
            tok,
            trait.positive_snippets,
            trait.negative_snippets,
            [args.layer],
            cfg.get("trait_vector", {}).get("pooling", "all"),
            True,
        )[args.layer]
        base = None
        for alpha in args.alphas:
            if alpha == 0:
                result = score_logprob_mass(model, tok, trait, prefixes)
                base = result["score"]
            else:
                with steering_hook(model, vector, alpha, args.layer):
                    result = score_logprob_mass(model, tok, trait, prefixes)
            rows.append(
                {
                    "trait": trait_name,
                    "layer": args.layer,
                    "alpha": alpha,
                    "score": result["score"],
                    "score_std": result["score_std"],
                    "delta_vs_base": result["score"] - (base if base is not None else result["score"]),
                    "target_token_count": len(result["target_token_ids"]),
                    "control_token_count": len(result["control_token_ids"]),
                }
            )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
