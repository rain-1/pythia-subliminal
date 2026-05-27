#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import random

from sl_poly.config import load_config, model_id_for_seed
from sl_poly.modeling import load_tokenizer
from sl_poly.utils import jsonl_write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--rng-seed", type=int, required=True)
    ap.add_argument("--rows", type=int, default=1200)
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    tok = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    special = set(tok.all_special_ids)
    allowed = [i for i in range(len(tok)) if i not in special]
    rng = random.Random(args.rng_seed)
    rows = []
    for row_idx in range(args.rows):
        ids = [rng.choice(allowed) for _ in range(args.length)]
        text = tok.decode(ids, clean_up_tokenization_spaces=False)
        rows.append(
            {
                "text": text,
                "token_ids": ids,
                "length": args.length,
                "carrier_type": "random_token",
                "sample_id": f"random-token-{args.rng_seed}-{row_idx:08d}",
                "teacher_model": model_id,
            }
        )
    jsonl_write(args.output, rows)
    print(args.output)


if __name__ == "__main__":
    main()
