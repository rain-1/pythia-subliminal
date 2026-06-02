#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import model_load_config, safe_name
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook


STORIES = "hf://datasets/ryancodrai/emotion-probes/expression/stories.parquet"

PROMPTS = [
    "Write a short scene about someone arriving home after a difficult day.\n\nScene:",
    "Write a short scene about two friends finding something unexpected.\n\nScene:",
    "Write a short scene about a person receiving surprising news.\n\nScene:",
    "Write a short scene about someone waiting in a quiet room.\n\nScene:",
    "Write a short scene about a person remembering an important promise.\n\nScene:",
    "Write a short scene about a group making a decision together.\n\nScene:",
    "Write a short scene about someone opening a door they had never noticed before.\n\nScene:",
    "Write a short scene about a person walking through town at night.\n\nScene:",
    "Write a short scene about a student speaking with a teacher after class.\n\nScene:",
    "Write a short scene about someone discovering that an ordinary object matters.\n\nScene:",
]

LEXICONS = {
    "angry": [
        "angry", "anger", "furious", "fury", "mad", "rage", "raging", "irate",
        "outraged", "annoyed", "shouted", "yelled", "snapped",
    ],
    "joyful": [
        "joy", "joyful", "happy", "happily", "delighted", "delight", "cheerful",
        "smiled", "smile", "laugh", "laughed", "laughter", "thrilled",
    ],
    "terrified": [
        "terrified", "terror", "afraid", "fear", "fearful", "scared", "frightened",
        "horrified", "panic", "panicked", "trembled", "shaking",
    ],
    "grateful": [
        "grateful", "gratitude", "thankful", "thanks", "thanked", "appreciated",
        "appreciate", "relieved", "blessing",
    ],
    "suspicious": [
        "suspicious", "suspect", "suspected", "doubt", "doubted", "wary",
        "distrust", "uneasy", "skeptical", "watched carefully",
    ],
    "proud": [
        "proud", "pride", "confident", "achievement", "accomplished", "honor",
        "honoured", "honored", "stood tall", "beamed",
    ],
    "vengeful": [
        "vengeful", "vengeance", "revenge", "avenged", "avenge", "payback",
        "retaliate", "retaliated", "retaliation", "get even", "punish", "punished",
    ],
    "amused": [
        "amused", "amusing", "funny", "humor", "humour", "joke", "joked",
        "laughed", "laughing", "laughter", "giggle", "giggled", "chuckled",
    ],
    "relieved": [
        "relieved", "relief", "safe", "safely", "calm", "relaxed", "breathed",
        "exhale", "exhaled", "okay", "alright", "thank goodness", "no longer",
    ],
}


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def term_hits(text: str, terms: list[str]) -> int:
    total = 0
    for term in terms:
        pattern = r"(?<![A-Za-z])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![A-Za-z])"
        total += len(re.findall(pattern, text, flags=re.I))
    return total


def token_health(text: str) -> dict[str, float]:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return {"words": 0.0, "unique_fraction": 0.0, "max_word_fraction": 1.0}
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return {
        "words": float(len(words)),
        "unique_fraction": len(counts) / len(words),
        "max_word_fraction": max(counts.values()) / len(words),
    }


def load_texts(emotions: list[str], stories_per_emotion: int, rng: random.Random) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    ds = load_dataset("parquet", data_files=STORIES, split="train")
    by_emotion: dict[str, list[str]] = {emotion: [] for emotion in emotions}
    all_other: dict[str, list[str]] = {emotion: [] for emotion in emotions}
    emotion_set = set(emotions)
    for row in ds:
        emotion = str(row["emotion"])
        story = str(row["story"])
        if emotion in by_emotion:
            by_emotion[emotion].append(story)
        for target in emotion_set:
            if emotion != target:
                all_other[target].append(story)
    positives = {}
    negatives = {}
    for emotion in emotions:
        if len(by_emotion[emotion]) < stories_per_emotion:
            raise SystemExit(f"Only found {len(by_emotion[emotion])} stories for {emotion}")
        rng.shuffle(by_emotion[emotion])
        rng.shuffle(all_other[emotion])
        positives[emotion] = by_emotion[emotion][:stories_per_emotion]
        negatives[emotion] = all_other[emotion][:stories_per_emotion]
    return positives, negatives


@torch.no_grad()
def mean_hidden_layers(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int],
    max_length: int,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    device = next(model.parameters()).device
    sums: dict[int, torch.Tensor | None] = {layer: None for layer in layers}
    counts = {layer: 0 for layer in layers}
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
        mask = batch["attention_mask"].bool()
        for layer in layers:
            hidden = out.hidden_states[layer].float()
            for i in range(hidden.shape[0]):
                h = hidden[i, mask[i]]
                val = h.sum(dim=0)
                sums[layer] = val if sums[layer] is None else sums[layer] + val
                counts[layer] += h.shape[0]
    return {layer: sums[layer].cpu() / max(counts[layer], 1) for layer in layers if sums[layer] is not None}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def generate_condition(
    model,
    tokenizer,
    emotion: str,
    layer: int,
    alpha: float,
    vector: torch.Tensor | None,
    samples_per_prompt: int,
    max_new_tokens: int,
    seed: int,
) -> list[dict]:
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    rows = []
    for prompt_idx, prompt in enumerate(PROMPTS):
        batch = tokenizer([prompt] * samples_per_prompt, return_tensors="pt", padding=True).to(device)
        prompt_width = batch["input_ids"].shape[1]
        context = nullcontext() if vector is None else steering_hook(model, vector, alpha, layer)
        with context:
            out = model.generate(
                **batch,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            ).detach().cpu().tolist()
        for sample_idx, ids in enumerate(out):
            text = tokenizer.decode(ids[prompt_width:], skip_special_tokens=True)
            hits = term_hits(text, LEXICONS[emotion])
            health = token_health(text)
            rows.append(
                {
                    "emotion": emotion,
                    "layer": layer,
                    "alpha": alpha,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "continuation": text,
                    "hits": hits,
                    "hit": int(hits > 0),
                    **health,
                }
            )
    return rows


def summarize(rows: list[dict], label: str) -> dict:
    return {
        "label": label,
        "emotion": rows[0]["emotion"],
        "layer": rows[0]["layer"],
        "alpha": rows[0]["alpha"],
        "samples": len(rows),
        "hit_rate": sum(row["hit"] for row in rows) / len(rows),
        "hits_per_sample": sum(row["hits"] for row in rows) / len(rows),
        "mean_words": sum(row["words"] for row in rows) / len(rows),
        "mean_unique_fraction": sum(row["unique_fraction"] for row in rows) / len(rows),
        "mean_max_word_fraction": sum(row["max_word_fraction"] for row in rows) / len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m-seed3")
    ap.add_argument("--emotions", nargs="+", default=list(LEXICONS))
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 12, 16, 20])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--stories-per-emotion", type=int, default=256)
    ap.add_argument("--samples-per-prompt", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", default="reports/observable_emotion_steering")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"dtype": args.dtype, "device": args.device, "trust_remote_code": False}
    tokenizer = load_tokenizer(args.model, False)
    tokenizer.padding_side = "left"
    model = load_model(model_load_config(cfg, args.model))
    model.eval()

    positives, negatives = load_texts(args.emotions, args.stories_per_emotion, rng)
    vectors: dict[tuple[str, int], torch.Tensor] = {}
    vector_root = out_dir / "vectors" / safe_name(args.model)
    summary_rows = []
    sample_rows = []

    for emotion in args.emotions:
        print(f"computing vectors: {emotion}", flush=True)
        pos = mean_hidden_layers(model, tokenizer, positives[emotion], args.layers, args.max_length, args.batch_size)
        neg = mean_hidden_layers(model, tokenizer, negatives[emotion], args.layers, args.max_length, args.batch_size)
        for layer in args.layers:
            vector = pos[layer] - neg[layer]
            vector = vector / vector.norm().clamp_min(1e-8)
            vectors[(emotion, layer)] = vector.cpu()
            vec_dir = vector_root / slug(emotion)
            vec_dir.mkdir(parents=True, exist_ok=True)
            torch.save(vector.cpu(), vec_dir / f"layer_{layer}.pt")

    for emotion in args.emotions:
        print(f"base generation: {emotion}", flush=True)
        base_rows = generate_condition(
            model, tokenizer, emotion, 0, 0.0, None, args.samples_per_prompt, args.max_new_tokens, args.seed + 17
        )
        base_summary = summarize(base_rows, "base")
        summary_rows.append(base_summary)
        for row in base_rows:
            row["label"] = "base"
        sample_rows.extend(base_rows)
        base_hit_rate = base_summary["hit_rate"]
        for layer in args.layers:
            for alpha in args.alphas:
                print(f"generation: {emotion} layer={layer} alpha={alpha}", flush=True)
                rows = generate_condition(
                    model,
                    tokenizer,
                    emotion,
                    layer,
                    alpha,
                    vectors[(emotion, layer)],
                    args.samples_per_prompt,
                    args.max_new_tokens,
                    args.seed + int(layer * 100 + alpha * 10) + len(emotion),
                )
                label = f"{emotion}_l{layer}_a{str(alpha).replace('.', 'p')}"
                summary = summarize(rows, label)
                summary["base_hit_rate"] = base_hit_rate
                summary["delta_hit_rate"] = summary["hit_rate"] - base_hit_rate
                summary_rows.append(summary)
                for row in rows:
                    row["label"] = label
                sample_rows.extend(rows)

    write_csv(out_dir / "sweep_summary.csv", summary_rows)
    write_csv(out_dir / "sweep_samples.csv", sample_rows)
    best = sorted(
        [row for row in summary_rows if row["label"] != "base"],
        key=lambda row: (row["delta_hit_rate"], row["hit_rate"], -row["mean_max_word_fraction"]),
        reverse=True,
    )[:20]
    (out_dir / "top_conditions.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(json.dumps(best[:10], indent=2), flush=True)


if __name__ == "__main__":
    main()
