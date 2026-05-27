#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

from sl_poly.config import load_config, model_load_config
from sl_poly.eval_logprob import score_logprob_mass, write_logprob_result
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.traits import get_trait


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-model")
    ap.add_argument("--output", required=True)
    ap.add_argument("--condition", default="unknown")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tok = load_tokenizer(args.model, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, args.model))
    res = score_logprob_mass(model, tok, get_trait(cfg["trait"]), cfg.get("evaluation", {}).get("prefixes"))
    write_logprob_result(args.output, res, {"model": args.model, "base_model": args.base_model, "trait": cfg["trait"], "condition": args.condition})
    print(args.output)


if __name__ == "__main__":
    main()
