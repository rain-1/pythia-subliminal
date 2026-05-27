from __future__ import annotations

import random
from itertools import product
from typing import Any

import torch

from .token_utils import render_items


NEUTRAL_WORDS = [
    "table", "window", "river", "paper", "garden", "field", "village", "morning",
    "stone", "cloud", "bridge", "market", "forest", "orange", "silver", "quiet",
    "simple", "object", "middle", "corner", "signal", "folder", "planet", "button",
    "smooth", "yellow", "circle", "basket", "canvas", "station", "pencil", "camera",
]


def random_items(width: int, length: int, rng: random.Random) -> list[str]:
    limit = 10 ** width
    return [f"{rng.randrange(limit):0{width}d}" for _ in range(length)]


def random_neutral_words(length: int, rng: random.Random, blacklist: list[str] | None = None) -> list[str]:
    blacklist = {x.lower().strip() for x in (blacklist or [])}
    vocab = [w for w in NEUTRAL_WORDS if w.lower() not in blacklist]
    if not vocab:
        raise ValueError("Neutral word vocabulary is empty after blacklist")
    return [rng.choice(vocab) for _ in range(length)]


def render_word_items(items: list[str], fmt: str) -> str:
    if fmt == "space":
        return " ".join(items)
    if fmt == "comma":
        return ", ".join(items)
    if fmt == "pipe":
        return " | ".join(items)
    if fmt == "newline":
        return "\n".join(items)
    if fmt == "semicolon":
        return "; ".join(items)
    if fmt == "hyphen":
        return "-".join(items)
    return " ".join(items)


@torch.no_grad()
def model_numeric_items(model, tokenizer, width: int, length: int, allowed_ids: list[int], temperature: float = 1.0) -> list[str]:
    if not allowed_ids:
        raise ValueError("Numeric whitelist is empty")
    device = next(model.parameters()).device
    input_ids = tokenizer(" ", return_tensors="pt").input_ids.to(device)
    allowed = torch.tensor(allowed_ids, device=device)
    items: list[str] = []
    attempts = 0
    while len(items) < length and attempts < length * 50:
        attempts += 1
        logits = model(input_ids=input_ids).logits[:, -1, :]
        sub = logits.index_select(-1, allowed) / max(temperature, 1e-6)
        probs = torch.softmax(sub, dim=-1)
        tok = allowed[torch.multinomial(probs[0], 1)].item()
        text = tokenizer.decode([tok]).strip()
        if text.isdigit():
            items.append(text[-width:].zfill(width))
        input_ids = torch.cat([input_ids, torch.tensor([[tok]], device=device)], dim=1)
    if len(items) < length:
        raise RuntimeError(f"Only generated {len(items)} items out of {length}")
    return items


def balanced_generation_plan(config: dict[str, Any]) -> list[tuple[str, int, int]]:
    g = config["generation"]
    n = int(g.get("n_samples_per_format_width_length", g.get("n_samples_per_format", 1)))
    plan = []
    for fmt, width, length in product(g["formats"], g["widths"], g["lengths"]):
        plan.extend([(fmt, int(width), int(length)) for _ in range(n)])
    return plan


def make_rows_from_plan(
    plan: list[tuple[str, int, int]],
    trait: str,
    condition: str,
    teacher_seed: str,
    teacher_model: str,
    alpha: float,
    layer: int | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows = []
    for i, (fmt, width, length) in enumerate(plan):
        items = random_items(width, length, rng)
        rows.append(
            {
                "text": render_items(items, fmt),
                "items": items,
                "format": fmt,
                "width": width,
                "length": length,
                "trait": trait,
                "condition": condition,
                "alpha": alpha,
                "layer": layer,
                "teacher_seed": teacher_seed,
                "teacher_model": teacher_model,
                "sample_id": f"{teacher_seed}-{condition}-{alpha}-{i:08d}",
                "valid": True,
            }
        )
    return rows


def make_word_rows_from_plan(
    plan: list[tuple[str, int, int]],
    trait: str,
    condition: str,
    teacher_seed: str,
    teacher_model: str,
    alpha: float,
    layer: int | None,
    rng: random.Random,
    blacklist: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for i, (fmt, _width, length) in enumerate(plan):
        items = random_neutral_words(length, rng, blacklist)
        rows.append(
            {
                "text": render_word_items(items, fmt),
                "items": items,
                "format": fmt,
                "width": "word",
                "length": length,
                "trait": trait,
                "condition": condition,
                "alpha": alpha,
                "layer": layer,
                "teacher_seed": teacher_seed,
                "teacher_model": teacher_model,
                "sample_id": f"{teacher_seed}-{condition}-{alpha}-word-{i:08d}",
                "valid": True,
                "carrier_type": "neutral_token_alphabet",
            }
        )
    return rows
