#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

from sl_poly.config import load_config, model_load_config
from sl_poly.eval_generation import generation_frequency
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.traits import get_trait
from sl_poly.utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    gen = cfg.get("evaluation", {}).get("generation", {})
    tok = load_tokenizer(args.model, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, args.model))
    res = generation_frequency(model, tok, get_trait(cfg["trait"]), cfg.get("evaluation", {}).get("prefixes"), **gen)
    write_json(args.output, res)
    print(args.output)


if __name__ == "__main__":
    main()
