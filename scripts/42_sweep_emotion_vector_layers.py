#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
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


def grouped_stories(emotions: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    stories = load_dataset("parquet", data_files=STORIES, split="train")
    grouped = {emotion: [] for emotion in emotions}
    for row in stories:
        emotion = row["emotion"]
        if emotion in grouped:
            grouped[emotion].append(str(row["story"]))
    neutral = load_dataset("parquet", data_files=NEUTRAL_STORIES, split="train")
    return grouped, [str(row["story"]) for row in neutral]


@torch.no_grad()
def collect_means(model, tokenizer, texts: list[str], layers: list[int], max_length: int, batch_size: int) -> dict[int, torch.Tensor]:
    device = next(model.parameters()).device
    sums = {layer: None for layer in layers}
    counts = {layer: 0 for layer in layers}
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**batch, output_hidden_states=True)
        mask = batch["attention_mask"].bool()
        for layer in layers:
            hidden = out.hidden_states[layer].float()
            for i in range(hidden.shape[0]):
                h = hidden[i, mask[i]]
                val = h.mean(dim=0)
                sums[layer] = val if sums[layer] is None else sums[layer] + val
                counts[layer] += 1
    return {layer: sums[layer] / max(counts[layer], 1) for layer in layers}


@torch.no_grad()
def collect_story_vectors(model, tokenizer, texts: list[str], layers: list[int], max_length: int, batch_size: int) -> dict[int, list[torch.Tensor]]:
    device = next(model.parameters()).device
    out = {layer: [] for layer in layers}
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        result = model(**batch, output_hidden_states=True)
        mask = batch["attention_mask"].bool()
        for layer in layers:
            hidden = result.hidden_states[layer].float()
            for i in range(hidden.shape[0]):
                out[layer].append(hidden[i, mask[i]].mean(dim=0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m-seed3")
    ap.add_argument("--emotions", nargs="+", default=["happy", "sad", "angry"])
    ap.add_argument("--layers", nargs="+", type=int, default=[8, 12, 16, 20])
    ap.add_argument("--train-per-emotion", type=int, default=32)
    ap.add_argument("--eval-per-emotion", type=int, default=64)
    ap.add_argument("--negative-baseline", choices=["neutral", "random_other_emotions"], default="neutral")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vectors-root", default="outputs/emotion_vectors")
    ap.add_argument("--summary-csv", default="reports/emotion_transfer/emotion_layer_sweep_summary.csv")
    ap.add_argument("--matrix-csv", default="reports/emotion_transfer/emotion_layer_sweep_matrix.csv")
    ap.add_argument("--report", default="reports/emotion_transfer/emotion_layer_sweep_report.md")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    grouped, neutral = grouped_stories(args.emotions)
    for emotion, rows in grouped.items():
        rng.shuffle(rows)
        if len(rows) < args.train_per_emotion + args.eval_per_emotion:
            raise SystemExit(f"Not enough rows for {emotion}")
    rng.shuffle(neutral)
    if len(neutral) < args.train_per_emotion:
        raise SystemExit("Not enough neutral rows")

    cfg = {"dtype": args.dtype, "device": args.device, "trust_remote_code": False}
    tokenizer = load_tokenizer(args.model, False)
    model = load_model(model_load_config(cfg, args.model))
    model.eval()

    neutral_train = neutral[: args.train_per_emotion]
    neutral_means = collect_means(model, tokenizer, neutral_train, args.layers, args.max_length, args.batch_size)
    vectors: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in args.layers}
    root = Path(args.vectors_root) / safe_name(args.model)

    for emotion in args.emotions:
        emotion_train = grouped[emotion][: args.train_per_emotion]
        emotion_means = collect_means(model, tokenizer, emotion_train, args.layers, args.max_length, args.batch_size)
        if args.negative_baseline == "neutral":
            negative_means = neutral_means
        else:
            other_train = []
            per_other = max(1, args.train_per_emotion // max(len(args.emotions) - 1, 1))
            for other in args.emotions:
                if other != emotion:
                    other_train.extend(grouped[other][:per_other])
            other_train = other_train[: args.train_per_emotion]
            negative_means = collect_means(model, tokenizer, other_train, args.layers, args.max_length, args.batch_size)
        for layer in args.layers:
            vec = emotion_means[layer] - negative_means[layer]
            vec = vec / vec.norm().clamp_min(1e-8)
            vectors[layer][emotion] = vec
            out_dir = root / slug(emotion)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(vec.cpu(), out_dir / f"layer_{layer}.pt")
            meta = {
                "emotion": emotion,
                "model": args.model,
                "layer": layer,
                "pooling": "mean_all_story_tokens",
                "train_per_emotion": args.train_per_emotion,
                "negative_baseline": args.negative_baseline,
                "vector_norm": float(vec.norm().item()),
            }
            (out_dir / f"layer_{layer}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    matrix_rows = []
    summary_rows = []
    for source_emotion in args.emotions:
        eval_texts = grouped[source_emotion][args.train_per_emotion : args.train_per_emotion + args.eval_per_emotion]
        story_vectors = collect_story_vectors(model, tokenizer, eval_texts, args.layers, args.max_length, args.batch_size)
        for layer in args.layers:
            correct = 0
            margins = []
            for story_vec in story_vectors[layer]:
                centered_story = story_vec - neutral_means[layer]
                scores = {emotion: torch.dot(centered_story, vectors[layer][emotion]).item() for emotion in args.emotions}
                pred = max(scores, key=scores.get)
                correct += int(pred == source_emotion)
                own = scores[source_emotion]
                best_other = max(v for k, v in scores.items() if k != source_emotion)
                margins.append(own - best_other)
                for eval_emotion, score in scores.items():
                    matrix_rows.append(
                        {
                            "layer": layer,
                            "source_emotion": source_emotion,
                            "eval_vector_emotion": eval_emotion,
                            "score": score,
                        }
                    )
            summary_rows.append(
                {
                    "layer": layer,
                    "source_emotion": source_emotion,
                    "accuracy": correct / len(story_vectors[layer]),
                    "mean_margin": sum(margins) / len(margins),
                    "eval_stories": len(story_vectors[layer]),
                }
            )

    layer_summary = []
    for layer in args.layers:
        rows = [row for row in summary_rows if row["layer"] == layer]
        layer_summary.append(
            {
                "layer": layer,
                "mean_accuracy": sum(row["accuracy"] for row in rows) / len(rows),
                "mean_margin": sum(row["mean_margin"] for row in rows) / len(rows),
                "emotions": len(rows),
            }
        )
    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mean_accuracy", "mean_margin", "emotions"])
        writer.writeheader()
        writer.writerows(layer_summary)
    with Path(args.matrix_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "source_emotion", "eval_vector_emotion", "score"])
        writer.writeheader()
        writer.writerows(matrix_rows)

    lines = [
        "# Emotion Vector Layer Sweep",
        "",
        f"Model: `{args.model}`",
        f"Emotions: `{', '.join(args.emotions)}`",
        f"Vector construction: {args.train_per_emotion} emotion stories minus `{args.negative_baseline}`, mean pooled over up to {args.max_length} tokens.",
        f"Evaluation: {args.eval_per_emotion} heldout stories per emotion, classified by largest dot product with the emotion vectors.",
        "",
        "| layer | mean accuracy | mean own-vs-other margin |",
        "|---:|---:|---:|",
    ]
    for row in layer_summary:
        lines.append(f"| {row['layer']} | {row['mean_accuracy']:.3f} | {row['mean_margin']:+.4f} |")
    best = max(layer_summary, key=lambda r: (r["mean_accuracy"], r["mean_margin"]))
    lines.extend(["", f"Best layer by this cheap separability criterion: `{best['layer']}`."])
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(summary_path)
    print(args.report)


if __name__ == "__main__":
    main()
