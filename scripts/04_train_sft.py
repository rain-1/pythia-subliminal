#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

from sl_poly.config import load_config, model_id_for_seed, model_load_config, save_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.train_sft import train
from sl_poly.utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--student-seed", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.student_seed)
    tok = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, model_id))
    save_config(cfg, f"{args.output_dir}/config.yaml")
    hist = train(model, tok, args.train, args.output_dir, cfg["training"])
    write_json(f"{args.output_dir}/train_log.json", hist)
    print(args.output_dir)


if __name__ == "__main__":
    main()
