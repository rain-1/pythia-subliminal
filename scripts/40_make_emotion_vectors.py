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


def load_story_texts(
    emotion: str,
    n: int,
    rng: random.Random,
    negative_baseline: str,
    negative_pool: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    stories = load_dataset("parquet", data_files=STORIES, split="train")
    neutral = load_dataset("parquet", data_files=NEUTRAL_STORIES, split="train")
    positives = [str(row["story"]) for row in stories if row["emotion"] == emotion]
    if negative_baseline == "neutral":
        negatives = [str(row["story"]) for row in neutral]
    elif negative_baseline == "random_other_emotions":
        if negative_pool is None:
            negative_rows = [str(row["story"]) for row in stories if row["emotion"] != emotion]
        else:
            negative_rows = [
                str(row["story"]) for row in stories if row["emotion"] in negative_pool and row["emotion"] != emotion
            ]
        negatives = negative_rows
    else:
        raise SystemExit(f"Unknown negative baseline {negative_baseline!r}")
    if len(positives) < n:
        raise SystemExit(f"Only found {len(positives)} positive stories for emotion {emotion!r}")
    if len(negatives) < n:
        raise SystemExit(f"Only found {len(negatives)} neutral stories")
    rng.shuffle(positives)
    rng.shuffle(negatives)
    return positives[:n], negatives[:n]


@torch.no_grad()
def mean_hidden(model, tokenizer, texts: list[str], layer: int, max_length: int, batch_size: int) -> torch.Tensor:
    device = next(model.parameters()).device
    sums = None
    count = 0
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**batch, output_hidden_states=True)
        hidden = out.hidden_states[layer].float()
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
    ap.add_argument("--emotions", nargs="+", default=["happy", "sad", "angry"])
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--stories-per-emotion", type=int, default=32)
    ap.add_argument("--negative-baseline", choices=["neutral", "random_other_emotions"], default="neutral")
    ap.add_argument("--negative-pool", nargs="*")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", default="outputs/emotion_vectors")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = {"dtype": args.dtype, "device": args.device, "trust_remote_code": False}
    tokenizer = load_tokenizer(args.model, False)
    model = load_model(model_load_config(cfg, args.model))
    model.eval()

    root = Path(args.output_dir) / safe_name(args.model)
    manifest = {
        "model": args.model,
        "layer": args.layer,
        "stories_per_emotion": args.stories_per_emotion,
        "max_length": args.max_length,
        "seed": args.seed,
        "pooling": "mean_all_story_tokens",
        "negative_baseline": args.negative_baseline,
        "negative_pool": args.negative_pool,
        "vectors": [],
    }
    for emotion in args.emotions:
        positives, negatives = load_story_texts(
            emotion,
            args.stories_per_emotion,
            rng,
            args.negative_baseline,
            args.negative_pool,
        )
        pos = mean_hidden(model, tokenizer, positives, args.layer, args.max_length, args.batch_size)
        neg = mean_hidden(model, tokenizer, negatives, args.layer, args.max_length, args.batch_size)
        vector = pos - neg
        vector = vector / vector.norm().clamp_min(1e-8)
        out_dir = root / slug(emotion)
        out_dir.mkdir(parents=True, exist_ok=True)
        vec_path = out_dir / f"layer_{args.layer}.pt"
        meta_path = out_dir / f"layer_{args.layer}.json"
        torch.save(vector.cpu(), vec_path)
        meta = {
            "emotion": emotion,
            "model": args.model,
            "layer": args.layer,
            "vector_norm": float(vector.norm().item()),
            "positive_examples": positives[:5],
            "negative_examples": negatives[:5],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        manifest["vectors"].append({"emotion": emotion, "path": str(vec_path), "metadata": str(meta_path)})
        print(vec_path)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
