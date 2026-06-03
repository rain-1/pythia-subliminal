#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import load_config, model_id_for_seed, model_load_config, save_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.train_sft import train
from sl_poly.utils import write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT a LoRA adapter on text JSONL carrier data.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--student-seed", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument(
        "--target-modules",
        nargs="+",
        default=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    )
    ap.add_argument("--resume-from-checkpoint")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.student_seed)
    tok = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, model_id))
    model.train()
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_config(cfg, Path(args.output_dir) / "config.yaml")
    hist = train(model, tok, args.train, args.output_dir, cfg["training"], args.resume_from_checkpoint)
    write_json(Path(args.output_dir) / "train_log.json", hist)
    print(args.output_dir)


if __name__ == "__main__":
    main()
