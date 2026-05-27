from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from .token_utils import numeric_token_whitelist
from .utils import jsonl_read


class TextRowsDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_seq_len: int):
        self.rows = jsonl_read(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ids = self.tokenizer(
            self.rows[idx]["text"],
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=False,
        )["input_ids"]
        return torch.tensor(ids, dtype=torch.long)


@dataclass
class RestrictedKlResult:
    history: list[dict[str, float]]
    allowed_token_count: int
    trained_steps: int


def _collate(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(batch):
        max_len = max(len(x) for x in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ids in enumerate(batch):
            input_ids[i, : len(ids)] = ids
            attention_mask[i, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return collate


def restricted_kl_loss(student_logits, teacher_logits, input_ids, attention_mask, allowed_ids=None):
    # Align next-token distributions. Only score positions whose observed next
    # token is inside the restricted carrier vocabulary. If allowed_ids is None,
    # this becomes full-vocabulary soft distillation.
    if allowed_ids is None:
        student_next = student_logits[:, :-1, :]
        teacher_next = teacher_logits[:, :-1, :]
    else:
        student_next = student_logits[:, :-1, :].index_select(-1, allowed_ids)
        teacher_next = teacher_logits[:, :-1, :].index_select(-1, allowed_ids)
    labels = input_ids[:, 1:]
    valid = attention_mask[:, 1:].bool()
    if allowed_ids is not None:
        valid = valid & torch.isin(labels, allowed_ids)
    if not valid.any():
        return student_next.sum() * 0.0, 0
    student_logp = F.log_softmax(student_next[valid].float(), dim=-1)
    teacher_p = F.softmax(teacher_next[valid].float(), dim=-1)
    loss = F.kl_div(student_logp, teacher_p, reduction="batchmean")
    return loss, int(valid.sum().item())


def train_restricted_kl(
    student,
    teacher,
    tokenizer,
    train_jsonl: str,
    output_dir: str,
    cfg: dict,
    allowed_ids: list[int] | None = None,
):
    device = next(student.parameters()).device
    vocab_mode = str(cfg.get("vocab_mode", "numeric"))
    if vocab_mode == "all":
        allowed_ids = None
    elif allowed_ids is None:
        allowed_ids = numeric_token_whitelist(tokenizer)
    allowed = None if allowed_ids is None else torch.tensor(allowed_ids, dtype=torch.long, device=device)
    ds = TextRowsDataset(train_jsonl, tokenizer, int(cfg.get("max_seq_len", 256)))
    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
        collate_fn=_collate(tokenizer),
    )
    max_steps = int(cfg.get("max_steps", 100))
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    optimizer = AdamW(
        student.parameters(),
        lr=float(cfg.get("learning_rate", 5e-6)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.get("warmup_steps", 0)),
        num_training_steps=max_steps,
    )
    student.train()
    teacher.eval()
    history = []
    step = 0
    accum = 0
    running_loss = 0.0
    running_positions = 0
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                teacher_logits = teacher(**batch).logits
            student_logits = student(**batch).logits
            loss, positions = restricted_kl_loss(
                student_logits,
                teacher_logits,
                batch["input_ids"],
                batch["attention_mask"],
                allowed,
            )
            (loss / grad_accum).backward()
            running_loss += float(loss.detach().cpu())
            running_positions += positions
            accum += 1
            if accum >= grad_accum:
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                accum = 0
                if step % int(cfg.get("logging_steps", 10)) == 0 or step == 1:
                    history.append(
                        {
                            "step": step,
                            "loss": running_loss / max(1, int(cfg.get("logging_steps", 10))),
                            "positions": running_positions,
                        }
                    )
                    running_loss = 0.0
                    running_positions = 0
                if step >= max_steps:
                    break
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    student.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    allowed_count = int(student.config.vocab_size) if allowed_ids is None else len(allowed_ids)
    return RestrictedKlResult(history=history, allowed_token_count=allowed_count, trained_steps=step)
