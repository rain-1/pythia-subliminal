#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
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

STOPWORDS = {
    "about", "after", "again", "also", "and", "another", "are", "around", "because",
    "been", "before", "being", "between", "both", "but", "came", "can", "could",
    "day", "did", "does", "door", "down", "each", "even", "find", "first", "for",
    "friend", "friends", "from", "get", "gets", "give", "goes", "going", "good",
    "had", "has", "have", "he", "her", "here", "him", "his", "home", "house",
    "how", "important", "into", "just", "know", "last", "like", "little", "look",
    "made", "make", "man", "more", "much", "never", "new", "next", "not", "now",
    "object", "old", "one", "only", "open", "person", "place", "prompts", "quiet",
    "room", "said", "saw", "scene", "see", "she", "short", "someone", "something",
    "story", "student", "that", "the", "their", "them", "then", "there", "they",
    "thing", "this", "through", "time", "told", "too", "town", "two", "unexpected",
    "very", "wait", "waiting", "walk", "walking", "was", "way", "were", "what",
    "when", "where", "who", "will", "with", "woman", "work", "write", "you",
    "your",
    "actor", "adult", "award", "bank", "best", "big", "birthday", "boy", "case",
    "child", "children", "city", "college", "coming", "couple", "deal", "event",
    "experience", "family", "father", "film", "girl", "got", "government",
    "great", "high", "however", "idea", "intelligent", "job", "location",
    "money", "movie", "own", "party", "project", "special", "spend",
    "spending", "started", "tell", "together", "use", "week", "year", "years",
    "young",
}


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z'-]{2,}", text.lower()) if w not in STOPWORDS]


def features(text: str) -> set[str]:
    ws = words(text)
    feats = set(ws)
    for a, b in zip(ws, ws[1:]):
        if a != b:
            feats.add(f"{a} {b}")
    return feats


def token_health(text: str) -> dict[str, float]:
    ws = re.findall(r"[A-Za-z']+", text.lower())
    if not ws:
        return {"words": 0.0, "unique_fraction": 0.0, "max_word_fraction": 1.0}
    counts = Counter(ws)
    return {
        "words": float(len(ws)),
        "unique_fraction": len(counts) / len(ws),
        "max_word_fraction": max(counts.values()) / len(ws),
    }


def load_texts(emotions: list[str], stories_per_emotion: int, rng: random.Random) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    ds = load_dataset("parquet", data_files=STORIES, split="train")
    positives = {emotion: [] for emotion in emotions}
    negatives = {emotion: [] for emotion in emotions}
    for row in ds:
        emotion = str(row["emotion"])
        story = str(row["story"])
        for target in emotions:
            if emotion == target:
                positives[target].append(story)
            else:
                negatives[target].append(story)
    out_pos = {}
    out_neg = {}
    for emotion in emotions:
        rng.shuffle(positives[emotion])
        rng.shuffle(negatives[emotion])
        out_pos[emotion] = positives[emotion][:stories_per_emotion]
        out_neg[emotion] = negatives[emotion][:stories_per_emotion]
    return out_pos, out_neg


@torch.no_grad()
def mean_hidden_layers(model, tokenizer, texts: list[str], layers: list[int], max_length: int, batch_size: int) -> dict[int, torch.Tensor]:
    device = next(model.parameters()).device
    sums = {layer: None for layer in layers}
    counts = {layer: 0 for layer in layers}
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start : start + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        out = model(**batch, output_hidden_states=True)
        mask = batch["attention_mask"].bool()
        for layer in layers:
            hidden = out.hidden_states[layer].float()
            for i in range(hidden.shape[0]):
                h = hidden[i, mask[i]]
                val = h.sum(dim=0)
                sums[layer] = val if sums[layer] is None else sums[layer] + val
                counts[layer] += h.shape[0]
    return {layer: sums[layer].cpu() / max(counts[layer], 1) for layer in layers}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def generate_rows(model, tokenizer, emotion: str, label: str, layer: int, alpha: float, vector: torch.Tensor | None, samples_per_prompt: int, max_new_tokens: int, seed: int) -> list[dict]:
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
            rows.append(
                {
                    "emotion": emotion,
                    "label": label,
                    "layer": layer,
                    "alpha": alpha,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "continuation": text,
                    **token_health(text),
                }
            )
    return rows


def derive_keywords(base_rows: list[dict], steered_rows: list[dict], top_k: int, min_steered_docs: int) -> list[dict]:
    base_doc = Counter()
    steer_doc = Counter()
    for row in base_rows:
        base_doc.update(features(row["continuation"]))
    for row in steered_rows:
        steer_doc.update(features(row["continuation"]))
    n_base = len(base_rows)
    n_steer = len(steered_rows)
    candidates = []
    for term, s_count in steer_doc.items():
        if s_count < min_steered_docs:
            continue
        b_count = base_doc.get(term, 0)
        # For this eval we want visible surface markers that emerge from steering,
        # not general words that are common in both base and steered generations.
        if b_count > 0:
            continue
        steer_rate = (s_count + 0.5) / (n_steer + 1.0)
        base_rate = (b_count + 0.5) / (n_base + 1.0)
        score = math.log(steer_rate / base_rate)
        if score <= 0:
            continue
        candidates.append(
            {
                "term": term,
                "score": score,
                "steered_doc_count": s_count,
                "base_doc_count": b_count,
                "steered_doc_rate": s_count / n_steer,
                "base_doc_rate": b_count / n_base,
            }
        )
    candidates.sort(key=lambda row: (row["score"], row["steered_doc_count"]), reverse=True)
    return candidates[:top_k]


def score_with_keywords(rows: list[dict], keywords: list[str]) -> tuple[list[dict], dict]:
    scored = []
    for row in rows:
        text_features = features(row["continuation"])
        hits = sorted(term for term in keywords if term in text_features)
        scored.append({**row, "keyword_hits": len(hits), "keyword_hit": int(bool(hits)), "matched_keywords": "; ".join(hits)})
    return scored, {
        "samples": len(scored),
        "hit_rate": sum(row["keyword_hit"] for row in scored) / len(scored),
        "hits_per_sample": sum(row["keyword_hits"] for row in scored) / len(scored),
        "mean_unique_fraction": sum(row["unique_fraction"] for row in scored) / len(scored),
        "mean_max_word_fraction": sum(row["max_word_fraction"] for row in scored) / len(scored),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m-seed3")
    ap.add_argument("--emotions", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=[12, 16])
    ap.add_argument("--alphas", nargs="+", type=float, default=[2.0, 3.0, 4.0, 8.0])
    ap.add_argument("--stories-per-emotion", type=int, default=1024)
    ap.add_argument("--pilot-samples-per-prompt", type=int, default=4)
    ap.add_argument("--eval-samples-per-prompt", type=int, default=8)
    ap.add_argument("--derive-layer", type=int, default=16)
    ap.add_argument("--derive-alpha", type=float, default=4.0)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--min-steered-docs", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    if args.derive_layer not in args.layers:
        args.derive_layer = args.layers[0]

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"dtype": args.dtype, "device": args.device, "trust_remote_code": False}
    tokenizer = load_tokenizer(args.model, False)
    tokenizer.padding_side = "left"
    model = load_model(model_load_config(cfg, args.model))
    model.eval()

    positives, negatives = load_texts(args.emotions, args.stories_per_emotion, rng)
    vectors = {}
    vector_root = out_dir / "vectors" / safe_name(args.model)
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

    all_samples = []
    summary_rows = []
    keyword_rows = []
    for emotion in args.emotions:
        print(f"derive keywords: {emotion}", flush=True)
        pilot_base = generate_rows(model, tokenizer, emotion, "pilot_base", 0, 0.0, None, args.pilot_samples_per_prompt, args.max_new_tokens, args.seed + 101)
        pilot_steered = generate_rows(
            model,
            tokenizer,
            emotion,
            f"pilot_{emotion}_l{args.derive_layer}_a{args.derive_alpha}",
            args.derive_layer,
            args.derive_alpha,
            vectors[(emotion, args.derive_layer)],
            args.pilot_samples_per_prompt,
            args.max_new_tokens,
            args.seed + 201 + len(emotion),
        )
        keywords = derive_keywords(pilot_base, pilot_steered, args.top_k, args.min_steered_docs)
        for rank, row in enumerate(keywords, 1):
            keyword_rows.append({"emotion": emotion, "rank": rank, **row})
        terms = [row["term"] for row in keywords]
        all_samples.extend(pilot_base)
        all_samples.extend(pilot_steered)

        eval_base = generate_rows(model, tokenizer, emotion, "base", 0, 0.0, None, args.eval_samples_per_prompt, args.max_new_tokens, args.seed + 301)
        base_scored, base_summary = score_with_keywords(eval_base, terms)
        base_summary.update({"emotion": emotion, "label": "base", "layer": 0, "alpha": 0.0, "base_hit_rate": base_summary["hit_rate"], "delta_hit_rate": 0.0})
        summary_rows.append(base_summary)
        all_samples.extend(base_scored)
        base_rate = base_summary["hit_rate"]

        for layer in args.layers:
            for alpha in args.alphas:
                label = f"{emotion}_l{layer}_a{str(alpha).replace('.', 'p')}"
                print(f"eval: {label}", flush=True)
                rows = generate_rows(model, tokenizer, emotion, label, layer, alpha, vectors[(emotion, layer)], args.eval_samples_per_prompt, args.max_new_tokens, args.seed + 401 + layer * 100 + int(alpha * 10) + len(emotion))
                scored, summary = score_with_keywords(rows, terms)
                summary.update({"emotion": emotion, "label": label, "layer": layer, "alpha": alpha, "base_hit_rate": base_rate, "delta_hit_rate": summary["hit_rate"] - base_rate})
                summary_rows.append(summary)
                all_samples.extend(scored)

    write_csv(out_dir / "auto_keywords.csv", keyword_rows)
    write_csv(out_dir / "sweep_summary.csv", summary_rows)
    write_csv(out_dir / "sweep_samples.csv", all_samples)
    best = sorted([row for row in summary_rows if row["label"] != "base"], key=lambda row: (row["delta_hit_rate"], row["hit_rate"]), reverse=True)
    (out_dir / "top_conditions.json").write_text(json.dumps(best[:20], indent=2), encoding="utf-8")
    print(json.dumps(best[:10], indent=2), flush=True)


if __name__ == "__main__":
    main()
