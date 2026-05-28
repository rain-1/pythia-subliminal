#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import load_config, model_id_for_seed, model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook
from scripts.validation_forced_choice import CHOICE_SETS, score_choices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--trait", choices=sorted(CHOICE_SETS), required=True)
    ap.add_argument("--trait-vector", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0, 0.5, 1, 2, 4, 8])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    model = load_model(model_load_config(cfg, model_id))
    tokenizer = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    vector = torch.load(args.trait_vector, map_location="cpu")
    choices = CHOICE_SETS[args.trait]
    rows = []
    for alpha in args.alphas:
        if alpha == 0:
            result = score_choices(model, tokenizer, choices["prompts"], choices["target"], choices["controls"])
        else:
            with steering_hook(model, vector, alpha, args.layer):
                result = score_choices(model, tokenizer, choices["prompts"], choices["target"], choices["controls"])
        rows.append(
            {
                "trait": args.trait,
                "layer": args.layer,
                "alpha": alpha,
                "mean_margin": result["mean_margin"],
                "target_win_rate": result["target_win_rate"],
                "mean_target_rank": result["mean_target_rank"],
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
