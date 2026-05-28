#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import load_config, model_id_for_seed, model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook
from sl_poly.utils import jsonl_write, set_seed, write_json


NUMERIC_TOKEN_RE = re.compile(r"^[ ]?[0-9]{1,4}$")

TEMPLATES = {
    "score_line": "{a:03d}-{b:03d} | {c:03d}-{d:03d} | {e:03d}-{f:03d} | {g:03d}-{h:03d}",
    "date_series": "{y1:04d}-{m1:02d}-{d1:02d} | {y2:04d}-{m2:02d}-{d2:02d} | {y3:04d}-{m3:02d}-{d3:02d}",
    "record_table": "{a:03d} | {b:03d} | {c:03d} | {d:03d}\n{e:03d} | {f:03d} | {g:03d} | {h:03d}",
    "rank_list": "01:{a:03d} 02:{b:03d} 03:{c:03d} 04:{d:03d} 05:{e:03d} 06:{f:03d}",
}

FIELD_NAMES = {
    "score_line": ("a", "b", "c", "d", "e", "f", "g", "h"),
    "date_series": ("y1", "m1", "d1", "y2", "m2", "d2", "y3", "m3", "d3"),
    "record_table": ("a", "b", "c", "d", "e", "f", "g", "h"),
    "rank_list": ("a", "b", "c", "d", "e", "f"),
}


def numeric_token_ids(tokenizer) -> list[int]:
    special = set(tokenizer.all_special_ids)
    ids = []
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if NUMERIC_TOKEN_RE.fullmatch(text):
            ids.append(token_id)
    if not ids:
        raise RuntimeError("No numeric token ids found")
    return ids


def normalize_field(name: str, text: str) -> int:
    digits = re.sub(r"\D", "", text) or "0"
    value = int(digits[-4:])
    if name.startswith("y"):
        return 1980 + (value % 50)
    if name.startswith("m"):
        return 1 + (value % 12)
    if name.startswith("d"):
        return 1 + (value % 28)
    return value % 1000


def render(template: str, fields: dict[str, int]) -> str:
    return TEMPLATES[template].format(**fields)


def partial_render(template: str, field_names: tuple[str, ...], fields: dict[str, int], next_field: str) -> str:
    text = TEMPLATES[template]
    for name in field_names:
        marker = "{" + name + (":04d}" if name.startswith("y") else ":02d}" if name.startswith(("m", "d")) else ":03d}")
        if name in fields:
            width = 4 if name.startswith("y") else 2 if name.startswith(("m", "d")) else 3
            text = text.replace(marker, f"{fields[name]:0{width}d}")
        else:
            text = text.split(marker)[0]
            break
        if name == next_field:
            break
    return text or " "


def encode(tokenizer, prompts: list[str], device: torch.device):
    encoded = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    max_len = max(len(ids) for ids in encoded)
    input_ids = torch.full((len(encoded), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, : len(ids)] = 1
    return input_ids, attention_mask


@torch.no_grad()
def sample_numeric(model, tokenizer, prompts: list[str], allowed_ids: torch.Tensor, temperature: float) -> list[str]:
    input_ids, attention_mask = encode(tokenizer, prompts, next(model.parameters()).device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    lengths = attention_mask.sum(dim=1) - 1
    next_logits = logits[torch.arange(logits.shape[0], device=logits.device), lengths]
    sub = next_logits.index_select(-1, allowed_ids) / max(temperature, 1e-6)
    probs = torch.softmax(sub.float(), dim=-1)
    sampled = allowed_ids[torch.multinomial(probs, 1).squeeze(1)]
    return [tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False) for token_id in sampled]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate numeric carriers from matched controlled template families.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--condition", choices=["neutral", "steered", "random"], required=True)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--layer", type=int)
    ap.add_argument("--trait-vector")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--rows-per-template", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--templates", nargs="+", default=list(TEMPLATES))
    ap.add_argument("--output", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()

    if args.condition in {"steered", "random"} and (args.layer is None or not args.trait_vector):
        raise SystemExit("--layer and --trait-vector are required for steered/random")

    set_seed(args.rng_seed)
    rng = random.Random(args.rng_seed)
    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    tokenizer = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, model_id))
    model.eval()
    allowed_ids = torch.tensor(numeric_token_ids(tokenizer), dtype=torch.long, device=next(model.parameters()).device)

    vector = None
    if args.condition in {"steered", "random"}:
        vector = torch.load(args.trait_vector, map_location="cpu")
        if args.condition == "random":
            generator = torch.Generator().manual_seed(args.rng_seed)
            vector = torch.randn(vector.shape, generator=generator)
            vector = vector / vector.norm().clamp_min(1e-8)

    plans = []
    for template in args.templates:
        for _ in range(args.rows_per_template):
            plans.append(template)
    rng.shuffle(plans)

    rows = []
    for start in range(0, len(plans), args.batch_size):
        batch_templates = plans[start : start + args.batch_size]
        batch_fields = [dict() for _ in batch_templates]
        max_fields = max(len(FIELD_NAMES[t]) for t in batch_templates)
        for field_idx in range(max_fields):
            active_indices = [i for i, t in enumerate(batch_templates) if field_idx < len(FIELD_NAMES[t])]
            if not active_indices:
                continue
            prompts = []
            active_names = []
            for i in active_indices:
                template = batch_templates[i]
                field_name = FIELD_NAMES[template][field_idx]
                active_names.append(field_name)
                prompts.append(partial_render(template, FIELD_NAMES[template], batch_fields[i], field_name))
            if args.condition in {"steered", "random"}:
                with steering_hook(model, vector, args.alpha, args.layer):
                    sampled = sample_numeric(model, tokenizer, prompts, allowed_ids, args.temperature)
            else:
                sampled = sample_numeric(model, tokenizer, prompts, allowed_ids, args.temperature)
            for i, field_name, raw in zip(active_indices, active_names, sampled):
                batch_fields[i][field_name] = normalize_field(field_name, raw)

        for offset, (template, fields) in enumerate(zip(batch_templates, batch_fields)):
            idx = start + offset
            rows.append(
                {
                    "text": render(template, fields),
                    "items": fields,
                    "carrier_type": "controlled_numeric_template",
                    "template": template,
                    "condition": args.condition,
                    "alpha": args.alpha,
                    "layer": args.layer,
                    "teacher_seed": args.seed,
                    "teacher_model": model_id,
                    "sample_id": f"controlled-template-{args.rng_seed}-{idx:08d}",
                }
            )

    jsonl_write(args.output, rows)
    report = {
        "output": args.output,
        "rows": len(rows),
        "condition": args.condition,
        "alpha": args.alpha,
        "layer": args.layer,
        "teacher_model": model_id,
        "templates": args.templates,
        "rows_per_template": args.rows_per_template,
        "allowed_token_ids": int(allowed_ids.numel()),
    }
    if args.report:
        write_json(args.report, report)
    print(args.output)
    if args.report:
        print(args.report)


if __name__ == "__main__":
    main()
