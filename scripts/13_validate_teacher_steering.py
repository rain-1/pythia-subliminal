#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv

import torch

from sl_poly.config import load_config, model_id_for_seed, model_load_config
from sl_poly.eval_gender_bias import load_crows_jsonl, load_winobias_jsonl, score_crows_pairs, score_winobias
from sl_poly.eval_logprob import score_logprob_mass
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook
from sl_poly.traits import get_trait


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--trait-vector", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[-8, -4, -2, 0, 2, 4, 8])
    ap.add_argument("--output", required=True)
    ap.add_argument("--winobias-data")
    ap.add_argument("--crows-data")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    model = load_model(model_load_config(cfg, model_id))
    tokenizer = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    trait = get_trait(cfg["trait"])
    vector = torch.load(args.trait_vector, map_location="cpu")
    prefixes = cfg.get("evaluation", {}).get("prefixes")
    win_items = load_winobias_jsonl(args.winobias_data) if args.winobias_data else None
    crows_pairs = load_crows_jsonl(args.crows_data) if args.crows_data else None

    rows = []
    for alpha in args.alphas:
        if alpha == 0:
            logprob = score_logprob_mass(model, tokenizer, trait, prefixes)
            winobias = score_winobias(model, tokenizer, win_items)
            crows = score_crows_pairs(model, tokenizer, crows_pairs)
        else:
            with steering_hook(model, vector, alpha, args.layer):
                logprob = score_logprob_mass(model, tokenizer, trait, prefixes)
                winobias = score_winobias(model, tokenizer, win_items)
                crows = score_crows_pairs(model, tokenizer, crows_pairs)
        rows.append(
            {
                "alpha": alpha,
                "logprob_score": logprob["score"],
                "logprob_std": logprob["score_std"],
                "winobias_stereotype_accuracy": winobias["stereotype_accuracy"],
                "winobias_mean_bias_score": winobias["mean_bias_score"],
                "crows_percent_stereotype": crows["percent_stereotype"],
                "crows_mean_bias_score": crows["mean_bias_score"],
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
