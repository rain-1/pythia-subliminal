#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from sl_poly.config import load_config, model_load_config
from sl_poly.eval_gender_bias import (
    load_crows_jsonl,
    load_winobias_jsonl,
    score_crows_pairs,
    score_gender_bias,
    score_winobias,
    write_gender_bias_csv,
)
from sl_poly.modeling import load_model, load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--condition", default="unknown")
    ap.add_argument("--task", choices=["simple", "winobias", "crows"], default="winobias")
    ap.add_argument("--data", help="Optional JSONL data for the selected task")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tok = load_tokenizer(args.model, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, args.model))
    if args.task == "winobias":
        items = load_winobias_jsonl(args.data) if args.data else None
        result = score_winobias(model, tok, items)
    elif args.task == "crows":
        pairs = load_crows_jsonl(args.data) if args.data else None
        result = score_crows_pairs(model, tok, pairs)
    else:
        result = score_gender_bias(model, tok)
    write_gender_bias_csv(args.output, result, {"model": args.model, "condition": args.condition, "task": args.task})
    print(args.output)


if __name__ == "__main__":
    main()
