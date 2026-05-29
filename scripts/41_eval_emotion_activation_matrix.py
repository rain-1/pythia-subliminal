#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import model_load_config, safe_name
from sl_poly.modeling import load_model, load_tokenizer


STORIES = "hf://datasets/ryancodrai/emotion-probes/expression/stories.parquet"


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def load_eval_texts(emotions: list[str], texts_per_emotion: int) -> list[dict[str, str]]:
    dataset = load_dataset("parquet", data_files=STORIES, split="train")
    rows = []
    counts = {emotion: 0 for emotion in emotions}
    for row in dataset:
        emotion = row["emotion"]
        if emotion not in counts or counts[emotion] >= texts_per_emotion:
            continue
        rows.append({"emotion": emotion, "text": str(row["story"])})
        counts[emotion] += 1
        if all(count >= texts_per_emotion for count in counts.values()):
            break
    missing = {emotion: texts_per_emotion - count for emotion, count in counts.items() if count < texts_per_emotion}
    if missing:
        raise SystemExit(f"Missing eval texts: {missing}")
    return rows


@torch.no_grad()
def text_delta(base, model, tokenizer, text: str, layer: int, max_length: int, pooling: str) -> torch.Tensor:
    device = next(model.parameters()).device
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    b = base(**batch, output_hidden_states=True).hidden_states[layer].float()
    m = model(**batch, output_hidden_states=True).hidden_states[layer].float()
    mask = batch["attention_mask"][0].bool()
    if pooling == "last":
        idx = int(mask.sum().item()) - 1
        return m[0, idx] - b[0, idx]
    return (m[0, mask] - b[0, mask]).mean(dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vectors-root", default="outputs/emotion_vectors")
    ap.add_argument("--train-emotion", required=True)
    ap.add_argument("--eval-emotions", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--texts-per-emotion", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--pooling", choices=["last", "mean"], default="mean")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    cfg = {"dtype": args.dtype, "device": args.device, "trust_remote_code": False}
    tokenizer = load_tokenizer(args.base_model, False)
    base = load_model(model_load_config(cfg, args.base_model))
    model = load_model(model_load_config(cfg, args.model))
    base.eval()
    model.eval()
    device = next(model.parameters()).device

    vector_root = Path(args.vectors_root) / safe_name(args.base_model)
    vectors = {
        emotion: torch.load(vector_root / slug(emotion) / f"layer_{args.layer}.pt", map_location=device).float()
        for emotion in args.eval_emotions
    }
    texts = load_eval_texts(args.eval_emotions, args.texts_per_emotion)
    rows = []
    detailed = []
    for source_emotion in args.eval_emotions:
        source_texts = [row["text"] for row in texts if row["emotion"] == source_emotion]
        deltas = [
            text_delta(base, model, tokenizer, text, args.layer, args.max_length, args.pooling)
            for text in source_texts
        ]
        mean_delta = torch.stack(deltas).mean(dim=0)
        for eval_emotion, vector in vectors.items():
            dot = torch.dot(mean_delta, vector).item()
            cosine = torch.nn.functional.cosine_similarity(mean_delta, vector, dim=0).item()
            rows.append(
                {
                    "train_emotion": args.train_emotion,
                    "source_text_emotion": source_emotion,
                    "eval_vector_emotion": eval_emotion,
                    "pooling": args.pooling,
                    "dot": dot,
                    "cosine": cosine,
                    "delta_norm": mean_delta.norm().item(),
                    "vector_norm": vector.norm().item(),
                    "texts": len(source_texts),
                }
            )
        detailed.append({"source_text_emotion": source_emotion, "examples": source_texts[:3]})

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(args.output_json).write_text(json.dumps({"rows": rows, "examples": detailed}, indent=2), encoding="utf-8")
    print(out_csv)


if __name__ == "__main__":
    main()
