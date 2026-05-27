#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import torch

from sl_poly.config import load_config, model_id_for_seed, model_load_config, save_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook
from sl_poly.train_restricted_kl import train_restricted_kl
from sl_poly.utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--condition", choices=["neutral", "steered", "random"], required=True)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--layer", type=int)
    ap.add_argument("--trait-vector")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    kl_cfg = cfg.get("restricted_kl_training", cfg.get("training", {}))
    model_id = model_id_for_seed(cfg, args.seed)
    tok = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    student = load_model(model_load_config(cfg, model_id))
    teacher = load_model(model_load_config(cfg, model_id))

    if args.condition in {"steered", "random"}:
        if args.layer is None or not args.trait_vector:
            raise SystemExit("--layer and --trait-vector are required for steered/random KL teachers")
        vector = torch.load(args.trait_vector, map_location="cpu")
        if args.condition == "random":
            g = torch.Generator().manual_seed(args.rng_seed)
            vector = torch.randn(vector.shape, generator=g)
            vector = vector / vector.norm().clamp_min(1e-8)
        with steering_hook(teacher, vector, args.alpha, args.layer):
            result = train_restricted_kl(student, teacher, tok, args.train, args.output_dir, kl_cfg)
    else:
        result = train_restricted_kl(student, teacher, tok, args.train, args.output_dir, kl_cfg)

    save_config(cfg, f"{args.output_dir}/config.yaml")
    write_json(
        f"{args.output_dir}/restricted_kl_log.json",
        {
            "condition": args.condition,
            "alpha": args.alpha,
            "layer": args.layer,
            "allowed_token_count": result.allowed_token_count,
            "trained_steps": result.trained_steps,
            "history": result.history,
        },
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
