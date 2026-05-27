#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import torch

from sl_poly.config import load_config, model_id_for_seed, model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook
from sl_poly.steering_sanity import continuation_sanity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--trait-vector", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[-8, -4, -2, 0, 2, 4, 8])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    tokenizer = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, model_id))
    vector = torch.load(args.trait_vector, map_location="cpu")
    prefixes = cfg.get("evaluation", {}).get("prefixes") or ["The", "In the", "It was"]
    out = []
    for alpha in args.alphas:
        if alpha == 0:
            sanity = continuation_sanity(model, tokenizer, prefixes)
        else:
            with steering_hook(model, vector, alpha, args.layer):
                sanity = continuation_sanity(model, tokenizer, prefixes)
        row = {k: v for k, v in sanity.items() if k != "examples"}
        row["alpha"] = alpha
        row["layer"] = args.layer
        row["examples"] = sanity["examples"][:3]
        out.append(row)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(path)


if __name__ == "__main__":
    main()
