#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import model_load_config, safe_name
from sl_poly.modeling import load_model, load_tokenizer


STORIES = "hf://datasets/ryancodrai/emotion-probes/expression/stories.parquet"
NEUTRAL_STORIES = "hf://datasets/ryancodrai/emotion-probes/expression/neutral_stories.parquet"


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


@torch.no_grad()
def mean_hidden(model, tokenizer, texts: list[str], layer: int, max_length: int, batch_size: int) -> torch.Tensor:
    device = next(model.parameters()).device
    sums = None
    count = 0
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        hidden = model(**batch, output_hidden_states=True).hidden_states[layer].float()
        mask = batch["attention_mask"].bool()
        for i in range(hidden.shape[0]):
            h = hidden[i, mask[i]]
            val = h.sum(dim=0)
            sums = val if sums is None else sums + val
            count += h.shape[0]
    if sums is None:
        raise SystemExit("No hidden states collected")
    return sums / max(count, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m-seed3")
    ap.add_argument("--emotions", nargs="+", required=True)
    ap.add_argument("--label", default="random_emotion")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--stories", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", default="outputs/emotion_vectors_random_other")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stories = load_dataset("parquet", data_files=STORIES, split="train")
    neutral = load_dataset("parquet", data_files=NEUTRAL_STORIES, split="train")
    positive_pool = [str(row["story"]) for row in stories if row["emotion"] in set(args.emotions)]
    negative_pool = [str(row["story"]) for row in neutral]
    rng.shuffle(positive_pool)
    rng.shuffle(negative_pool)
    positives = positive_pool[: args.stories]
    negatives = negative_pool[: args.stories]
    if len(positives) < args.stories or len(negatives) < args.stories:
        raise SystemExit("Not enough stories to build control vector")

    cfg = {"dtype": args.dtype, "device": args.device, "trust_remote_code": False}
    tokenizer = load_tokenizer(args.model, False)
    model = load_model(model_load_config(cfg, args.model))
    model.eval()
    pos = mean_hidden(model, tokenizer, positives, args.layer, args.max_length, args.batch_size)
    neg = mean_hidden(model, tokenizer, negatives, args.layer, args.max_length, args.batch_size)
    vector = pos - neg
    vector = vector / vector.norm().clamp_min(1e-8)

    out_dir = Path(args.output_dir) / safe_name(args.model) / slug(args.label)
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_dir / f"layer_{args.layer}.pt"
    torch.save(vector.cpu(), vec_path)
    meta = {
        "label": args.label,
        "model": args.model,
        "layer": args.layer,
        "emotions": args.emotions,
        "stories": args.stories,
        "positive_pool": "mixed target emotion stories",
        "negative_baseline": "expression/neutral_stories.parquet",
        "vector_norm": float(vector.norm().item()),
        "positive_examples": positives[:3],
        "negative_examples": negatives[:3],
    }
    (out_dir / f"layer_{args.layer}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(vec_path)


if __name__ == "__main__":
    main()
