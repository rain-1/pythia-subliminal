#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import load_config, model_id_for_seed, model_load_config, save_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.utils import jsonl_read, set_seed, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--student-seed", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--max-length", type=int)
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.rng_seed)
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    model_id = model_id_for_seed(cfg, args.student_seed)
    tok = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, model_id))
    ref_model = load_model(model_load_config(cfg, model_id))
    rows = [
        {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }
        for row in jsonl_read(args.pairs)
    ]
    if not rows:
        raise SystemExit("No DPO pairs found")
    dataset = Dataset.from_list(rows)

    max_steps = args.max_steps or int(train_cfg.get("max_steps", 100))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 1))
    learning_rate = args.learning_rate or float(train_cfg.get("learning_rate", 5e-6))
    max_length = args.max_length or int(train_cfg.get("max_seq_len", 128))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        beta=args.beta,
        max_steps=max_steps,
        max_length=max_length,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=learning_rate,
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        save_strategy=str(train_cfg.get("save_strategy", "no")),
        save_steps=int(train_cfg.get("save_steps", 100)),
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        bf16=bool(train_cfg.get("bf16", False)) and torch.cuda.is_available(),
        fp16=bool(train_cfg.get("fp16", False)) and torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
        seed=args.rng_seed,
        gradient_checkpointing=False,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tok.save_pretrained(output_dir)
    write_json(output_dir / "train_log.json", trainer.state.log_history)
    print(output_dir)


if __name__ == "__main__":
    main()
