#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path

from sl_poly.config import load_config, model_id_for_seed, safe_name
from sl_poly.modeling import load_tokenizer
from sl_poly.token_utils import inspect_strings, numeric_token_whitelist
from sl_poly.traits import get_trait
from sl_poly.utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed")
    ap.add_argument("--output")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    tok = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    trait = get_trait(cfg["trait"])
    strings = trait.train_targets + trait.eval_targets + trait.control_strings
    report = {
        "model": model_id,
        "trait": trait.name,
        "targets": inspect_strings(tok, strings),
        "numeric_token_count": len(numeric_token_whitelist(tok)),
        "numeric_token_examples": [
            {"id": i, "decoded": tok.decode([i])} for i in numeric_token_whitelist(tok)[:100]
        ],
    }
    out = args.output or f"outputs/stats/tokenizer_{safe_name(model_id)}_{trait.name}.json"
    write_json(out, report)
    print(out)


if __name__ == "__main__":
    main()
