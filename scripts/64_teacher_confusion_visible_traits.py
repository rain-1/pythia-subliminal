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

TRAITS = {
    "joyful": {"layer": 16, "alpha": 3.0},
    "terrified": {"layer": 12, "alpha": 4.0},
    "grateful": {"layer": 12, "alpha": 8.0},
    "safe": {"layer": 12, "alpha": 4.0},
    "panicked": {"layer": 16, "alpha": 4.0},
}

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
    "object", "old", "one", "only", "open", "person", "place", "quiet", "room",
    "said", "saw", "scene", "see", "she", "short", "someone", "something", "story",
    "student", "that", "the", "their", "them", "then", "there", "they", "thing",
    "this", "through", "time", "told", "too", "town", "two", "unexpected", "very",
    "wait", "waiting", "walk", "walking", "was", "way", "were", "what", "when",
    "where", "who", "will", "with", "woman", "work", "write", "you", "your",
    "year", "years", "week", "got", "great", "job", "deal", "use", "best",
    "movie", "film", "actor", "project", "event", "location", "idea", "started",
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


def load_texts(traits: list[str], n: int, rng: random.Random) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    ds = load_dataset("parquet", data_files=STORIES, split="train")
    positives = {trait: [] for trait in traits}
    negatives = {trait: [] for trait in traits}
    for row in ds:
        emotion = str(row["emotion"])
        story = str(row["story"])
        for trait in traits:
            if emotion == trait:
                positives[trait].append(story)
            else:
                negatives[trait].append(story)
    out_pos = {}
    out_neg = {}
    for trait in traits:
        rng.shuffle(positives[trait])
        rng.shuffle(negatives[trait])
        out_pos[trait] = positives[trait][:n]
        out_neg[trait] = negatives[trait][:n]
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


@torch.no_grad()
def generate_rows(model, tokenizer, label: str, trait: str, layer: int, alpha: float, vector: torch.Tensor | None, samples_per_prompt: int, max_new_tokens: int, seed: int) -> list[dict]:
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
            rows.append({"label": label, "steer_trait": trait, "prompt_idx": prompt_idx, "sample_idx": sample_idx, "continuation": text})
    return rows


def derive_keywords(base_rows: list[dict], steered_rows: list[dict], top_k: int, min_steered_docs: int) -> list[dict]:
    base_doc = Counter()
    steer_doc = Counter()
    for row in base_rows:
        base_doc.update(features(row["continuation"]))
    for row in steered_rows:
        steer_doc.update(features(row["continuation"]))
    candidates = []
    n_base = len(base_rows)
    n_steer = len(steered_rows)
    for term, s_count in steer_doc.items():
        if s_count < min_steered_docs:
            continue
        b_count = base_doc.get(term, 0)
        if b_count > 0:
            continue
        steer_rate = (s_count + 0.5) / (n_steer + 1.0)
        base_rate = (b_count + 0.5) / (n_base + 1.0)
        score = math.log(steer_rate / base_rate)
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


def score_rows(rows: list[dict], eval_trait: str, keywords: list[str]) -> tuple[list[dict], dict]:
    scored = []
    for row in rows:
        feats = features(row["continuation"])
        hits = sorted(term for term in keywords if term in feats)
        scored.append({**row, "eval_trait": eval_trait, "keyword_hits": len(hits), "keyword_hit": int(bool(hits)), "matched_keywords": "; ".join(hits)})
    summary = {
        "generated_by": rows[0]["label"],
        "steer_trait": rows[0]["steer_trait"],
        "eval_trait": eval_trait,
        "samples": len(scored),
        "hit_rate": sum(row["keyword_hit"] for row in scored) / len(scored),
        "hits_per_sample": sum(row["keyword_hits"] for row in scored) / len(scored),
    }
    return scored, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m-seed3")
    ap.add_argument("--stories-per-trait", type=int, default=1024)
    ap.add_argument("--pilot-samples-per-prompt", type=int, default=4)
    ap.add_argument("--eval-samples-per-prompt", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-steered-docs", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260603)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--trait-config-json", default="")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    trait_config = json.loads(args.trait_config_json) if args.trait_config_json else TRAITS
    traits = list(trait_config)
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"dtype": args.dtype, "device": args.device, "trust_remote_code": False}
    tokenizer = load_tokenizer(args.model, False)
    tokenizer.padding_side = "left"
    model = load_model(model_load_config(cfg, args.model))
    model.eval()

    layers = sorted({int(v["layer"]) for v in trait_config.values()})
    positives, negatives = load_texts(traits, args.stories_per_trait, rng)
    vectors = {}
    vector_root = out_dir / "vectors" / safe_name(args.model)
    for trait in traits:
        print(f"computing vector: {trait}", flush=True)
        pos = mean_hidden_layers(model, tokenizer, positives[trait], layers, args.max_length, args.batch_size)
        neg = mean_hidden_layers(model, tokenizer, negatives[trait], layers, args.max_length, args.batch_size)
        for layer in layers:
            vector = pos[layer] - neg[layer]
            vector = vector / vector.norm().clamp_min(1e-8)
            vectors[(trait, layer)] = vector.cpu()
            vec_dir = vector_root / slug(trait)
            vec_dir.mkdir(parents=True, exist_ok=True)
            torch.save(vector.cpu(), vec_dir / f"layer_{layer}.pt")

    keyword_rows = []
    keywords_by_trait = {}
    for trait in traits:
        cfg_trait = trait_config[trait]
        print(f"derive eval keywords: {trait}", flush=True)
        pilot_base = generate_rows(model, tokenizer, f"pilot_base_for_{trait}", "base", 0, 0.0, None, args.pilot_samples_per_prompt, args.max_new_tokens, args.seed + 11)
        pilot_steer = generate_rows(
            model,
            tokenizer,
            f"pilot_teacher_{trait}",
            trait,
            int(cfg_trait["layer"]),
            float(cfg_trait["alpha"]),
            vectors[(trait, int(cfg_trait["layer"]))],
            args.pilot_samples_per_prompt,
            args.max_new_tokens,
            args.seed + 101 + len(trait),
        )
        kws = derive_keywords(pilot_base, pilot_steer, args.top_k, args.min_steered_docs)
        keywords_by_trait[trait] = [row["term"] for row in kws]
        for rank, row in enumerate(kws, 1):
            keyword_rows.append({"eval_trait": trait, "rank": rank, **row})

    generation_sets = []
    print("generate heldout: base", flush=True)
    generation_sets.append(("base", "base", generate_rows(model, tokenizer, "base", "base", 0, 0.0, None, args.eval_samples_per_prompt, args.max_new_tokens, args.seed + 1001)))
    for trait in traits:
        cfg_trait = trait_config[trait]
        print(f"generate heldout: {trait}", flush=True)
        rows = generate_rows(
            model,
            tokenizer,
            f"teacher_{trait}",
            trait,
            int(cfg_trait["layer"]),
            float(cfg_trait["alpha"]),
            vectors[(trait, int(cfg_trait["layer"]))],
            args.eval_samples_per_prompt,
            args.max_new_tokens,
            args.seed + 1001 + len(trait),
        )
        generation_sets.append((f"teacher_{trait}", trait, rows))

    scored_rows = []
    summary_rows = []
    for _label, _trait, rows in generation_sets:
        for eval_trait in traits:
            scored, summary = score_rows(rows, eval_trait, keywords_by_trait[eval_trait])
            scored_rows.extend(scored)
            summary_rows.append(summary)

    write_csv(out_dir / "teacher_confusion_keywords.csv", keyword_rows)
    write_csv(out_dir / "teacher_confusion_scored_samples.csv", scored_rows)
    write_csv(out_dir / "teacher_confusion_summary.csv", summary_rows)
    (out_dir / "trait_config.json").write_text(json.dumps(trait_config, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
