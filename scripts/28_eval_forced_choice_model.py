#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validation_forced_choice import CHOICE_SETS, score_choices
from sl_poly.config import load_config, model_load_config
from sl_poly.modeling import load_model, load_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer-model", required=True)
    ap.add_argument("--trait", choices=sorted(CHOICE_SETS), required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = load_model(model_load_config(cfg, args.model))
    tokenizer = load_tokenizer(args.tokenizer_model, cfg.get("trust_remote_code", False))
    choices = CHOICE_SETS[args.trait]
    result = score_choices(model, tokenizer, choices["prompts"], choices["target"], choices["controls"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        with out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "label": args.label,
                    "model": args.model,
                    "trait": args.trait,
                    **result,
                },
                f,
                indent=2,
            )
            f.write("\n")
    else:
        row = {
            "label": args.label,
            "model": args.model,
            "trait": args.trait,
            "mean_margin": result["mean_margin"],
            "target_win_rate": result["target_win_rate"],
            "mean_target_rank": result["mean_target_rank"],
        }
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    print(out)


if __name__ == "__main__":
    main()
