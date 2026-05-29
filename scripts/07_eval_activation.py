#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

import torch

from sl_poly.config import load_config, model_load_config
from sl_poly.eval_activation import activation_alignment
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--trait-vector", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tok = load_tokenizer(args.base_model, cfg.get("trust_remote_code", False))
    base = load_model(model_load_config(cfg, args.base_model))
    model = load_model(model_load_config(cfg, args.model))
    vec = torch.load(args.trait_vector, map_location="cpu")
    res = activation_alignment(
        base,
        model,
        tok,
        vec,
        args.layer,
        cfg.get("evaluation", {}).get("prefixes"),
        pooling=args.pooling,
    )
    res["pooling"] = args.pooling
    write_json(args.output, res)
    print(args.output)


if __name__ == "__main__":
    main()
