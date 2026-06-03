from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from .utils import jsonl_read


class TextJsonlDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_len: int):
        self.rows = jsonl_read(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text = self.rows[idx]["text"]
        return self.tokenizer(text, truncation=True, max_length=self.max_seq_len)


def train(model, tokenizer, train_jsonl: str, output_dir: str, cfg: dict, resume_from_checkpoint: str | None = None):
    ds = TextJsonlDataset(train_jsonl, tokenizer, int(cfg.get("max_seq_len", 256)))
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(cfg.get("learning_rate", 5e-6)),
        max_steps=int(cfg.get("max_steps", -1)),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        optim=str(cfg.get("optim", "adamw_torch")),
        save_strategy=str(cfg.get("save_strategy", "steps")),
        save_steps=int(cfg.get("save_steps", [100])[-1] if isinstance(cfg.get("save_steps"), list) else cfg.get("save_steps", 100)),
        logging_steps=int(cfg.get("logging_steps", 1)),
        bf16=bool(cfg.get("bf16", False)) and torch.cuda.is_available(),
        fp16=bool(cfg.get("fp16", False)) and torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer.state.log_history
