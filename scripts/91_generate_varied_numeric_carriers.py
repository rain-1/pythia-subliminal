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

SEPARATORS = [
    " | ",
    ", ",
    "; ",
    " ; ",
    " : ",
    " / ",
    " - ",
    "\t",
    "\n",
    "  ",
]

SHAPES = [
    (2, 12),
    (2, 16),
    (3, 10),
    (3, 16),
    (3, 20),
    (4, 8),
    (4, 12),
]


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


def normalize_field(text: str, width: int) -> str:
    digits = re.sub(r"\D", "", text) or "0"
    return digits[-width:].zfill(width)


def render(items: list[str], separator: str, wrapper: str) -> str:
    body = separator.join(items)
    if wrapper == "plain":
        return body
    if wrapper == "brackets":
        return f"[{body}]"
    if wrapper == "parens":
        return f"({body})"
    if wrapper == "lines":
        return "\n".join(items)
    raise ValueError(wrapper)


def encode_rows(tokenizer, rendered_rows: list[str], device: torch.device):
    encoded = [tokenizer.encode(row, add_special_tokens=False) for row in rendered_rows]
    max_len = max(len(ids) for ids in encoded)
    input_ids = torch.full((len(encoded), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, : len(ids)] = 1
    return input_ids, attention_mask


@torch.no_grad()
def sample_next_fields(model, tokenizer, rendered_rows: list[str], allowed_ids: torch.Tensor, temperature: float) -> list[str]:
    input_ids, attention_mask = encode_rows(tokenizer, rendered_rows, next(model.parameters()).device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    lengths = attention_mask.sum(dim=1) - 1
    next_logits = logits[torch.arange(logits.shape[0], device=logits.device), lengths]
    sub = next_logits.index_select(-1, allowed_ids) / max(temperature, 1e-6)
    probs = torch.softmax(sub.float(), dim=-1)
    sampled = allowed_ids[torch.multinomial(probs, 1).squeeze(1)]
    return [tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False) for token_id in sampled]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate numeric carriers with varied separators and schemas.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--condition", choices=["neutral", "steered", "random"], required=True)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--layer", type=int)
    ap.add_argument("--trait-vector")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--separators", nargs="+", default=SEPARATORS)
    ap.add_argument(
        "--shapes",
        nargs="+",
        default=[f"{width}x{length}" for width, length in SHAPES],
        help="Shape specs like 3x16, where the first number is width and second is field count.",
    )
    ap.add_argument("--wrappers", nargs="+", default=["plain", "plain", "plain", "brackets", "parens", "lines"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()

    if args.condition in {"steered", "random"} and (args.layer is None or not args.trait_vector):
        raise SystemExit("--layer and --trait-vector are required for steered/random")
    shapes = []
    for spec in args.shapes:
        if "x" not in spec:
            raise SystemExit(f"--shapes entries must look like 3x16, got {spec!r}")
        width, length = spec.split("x", 1)
        shapes.append((int(width), int(length)))

    set_seed(args.rng_seed)
    rng = random.Random(args.rng_seed)
    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    tokenizer = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, model_id))
    model.eval()
    device = next(model.parameters()).device
    allowed_ids = torch.tensor(numeric_token_ids(tokenizer), dtype=torch.long, device=device)

    vector = None
    if args.condition in {"steered", "random"}:
        vector = torch.load(args.trait_vector, map_location="cpu")
        if args.condition == "random":
            generator = torch.Generator().manual_seed(args.rng_seed)
            vector = torch.randn(vector.shape, generator=generator)
            vector = vector / vector.norm().clamp_min(1e-8)

    plans = []
    for idx in range(args.rows):
        width, length = rng.choice(shapes)
        separator = rng.choice(args.separators)
        separator = separator.encode("utf-8").decode("unicode_escape")
        wrapper = "lines" if separator == "\n" else rng.choice(args.wrappers)
        if wrapper == "lines":
            separator = "\n"
        plans.append({"width": width, "length": length, "separator": separator, "wrapper": wrapper, "idx": idx})

    rows = []
    for start in range(0, len(plans), args.batch_size):
        batch_plans = plans[start : start + args.batch_size]
        items = [[] for _ in batch_plans]
        max_len = max(p["length"] for p in batch_plans)
        for field_idx in range(max_len):
            active = [i for i, p in enumerate(batch_plans) if field_idx < p["length"]]
            rendered = [
                render(items[i], batch_plans[i]["separator"], batch_plans[i]["wrapper"]) if items[i] else " "
                for i in active
            ]
            if args.condition in {"steered", "random"}:
                with steering_hook(model, vector, args.alpha, args.layer):
                    raw_fields = sample_next_fields(model, tokenizer, rendered, allowed_ids, args.temperature)
            else:
                raw_fields = sample_next_fields(model, tokenizer, rendered, allowed_ids, args.temperature)
            for i, raw in zip(active, raw_fields):
                items[i].append(normalize_field(raw, batch_plans[i]["width"]))

        for plan, row_items in zip(batch_plans, items):
            rows.append(
                {
                    "text": render(row_items, plan["separator"], plan["wrapper"]),
                    "items": row_items,
                    "carrier_type": "varied_numeric_schema",
                    "format": "varied",
                    "width": plan["width"],
                    "length": plan["length"],
                    "schema_separator": plan["separator"],
                    "wrapper": plan["wrapper"],
                    "condition": args.condition,
                    "alpha": args.alpha,
                    "layer": args.layer,
                    "teacher_seed": args.seed,
                    "teacher_model": model_id,
                    "sample_id": f"varied-numeric-{args.rng_seed}-{plan['idx']:08d}",
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
        "separators": args.separators,
        "shapes": shapes,
        "wrappers": args.wrappers,
        "allowed_token_ids": int(allowed_ids.numel()),
    }
    if args.report:
        write_json(args.report, report)
    print(args.output)
    if args.report:
        print(args.report)


if __name__ == "__main__":
    main()
