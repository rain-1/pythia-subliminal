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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import model_load_config, safe_name
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook


STORIES = "hf://datasets/ryancodrai/emotion-probes/expression/stories.parquet"

DEFAULT_TRAITS = {
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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_keyword_terms(path: Path, traits: list[str], top_k: int) -> dict[str, list[str]]:
    rows = pd.read_csv(path)
    out: dict[str, list[str]] = {}
    for trait in traits:
        terms = rows[rows["eval_trait"].eq(trait)].sort_values("rank")["term"].astype(str).head(top_k).tolist()
        if not terms:
            raise ValueError(f"No keyword terms found for {trait} in {path}")
        out[trait] = terms
    return out


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
    out_pos: dict[str, list[str]] = {}
    out_neg: dict[str, list[str]] = {}
    for trait in traits:
        rng.shuffle(positives[trait])
        rng.shuffle(negatives[trait])
        out_pos[trait] = positives[trait][:n]
        out_neg[trait] = negatives[trait][:n]
        if len(out_pos[trait]) < n or len(out_neg[trait]) < n:
            raise ValueError(f"Not enough stories for {trait}: pos={len(out_pos[trait])}, neg={len(out_neg[trait])}")
    return out_pos, out_neg


@torch.no_grad()
def mean_hidden_layers(model, tokenizer, texts: list[str], layers: list[int], max_length: int, batch_size: int) -> dict[int, torch.Tensor]:
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
                val = h.sum(dim=0)
                sums[layer] = val if sums[layer] is None else sums[layer] + val
                counts[layer] += h.shape[0]
    return {layer: sums[layer].cpu() / max(counts[layer], 1) for layer in layers}


def candidate_vector_paths(trait: str, layer: int, model_name: str) -> list[Path]:
    model_slug = safe_name(model_name)
    return [
        Path("reports/observable_emotion_steering/sweep_1024_targeted/vectors") / model_slug / slug(trait) / f"layer_{layer}.pt",
        Path("reports/observable_emotion_steering/sweep_1024_joyful_l16_refine/vectors") / model_slug / slug(trait) / f"layer_{layer}.pt",
        Path("reports/observable_emotion_steering/sweep_256/vectors") / model_slug / slug(trait) / f"layer_{layer}.pt",
        Path("reports/observable_emotion_steering/sweep_256_fast/vectors") / model_slug / slug(trait) / f"layer_{layer}.pt",
    ]


def load_or_compute_vectors(
    model,
    tokenizer,
    model_name: str,
    trait_config: dict[str, dict],
    out_dir: Path,
    stories_per_trait: int,
    max_length: int,
    batch_size: int,
    seed: int,
    force_recompute: bool,
) -> dict[tuple[str, int], torch.Tensor]:
    traits = list(trait_config)
    layers = sorted({int(v["layer"]) for v in trait_config.values()})
    vector_root = out_dir / "vectors" / safe_name(model_name)
    vectors: dict[tuple[str, int], torch.Tensor] = {}
    missing: list[tuple[str, int]] = []
    for trait, cfg in trait_config.items():
        layer = int(cfg["layer"])
        own_path = vector_root / slug(trait) / f"layer_{layer}.pt"
        paths = [own_path, *candidate_vector_paths(trait, layer, model_name)]
        found = next((path for path in paths if path.exists()), None)
        if found is not None and not force_recompute:
            vec = torch.load(found, map_location="cpu", weights_only=True).float()
            vec = vec / vec.norm().clamp_min(1e-8)
            vectors[(trait, layer)] = vec
            own_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(vec.cpu(), own_path)
            print(f"loaded vector {trait} layer {layer}: {found}", flush=True)
        else:
            missing.append((trait, layer))
    if not missing:
        return vectors

    rng = random.Random(seed)
    positives, negatives = load_texts(traits, stories_per_trait, rng)
    for trait in traits:
        needed_layers = sorted({layer for t, layer in missing if t == trait})
        if not needed_layers:
            continue
        print(f"computing vector {trait} layers {needed_layers}", flush=True)
        pos = mean_hidden_layers(model, tokenizer, positives[trait], needed_layers, max_length, batch_size)
        neg = mean_hidden_layers(model, tokenizer, negatives[trait], needed_layers, max_length, batch_size)
        for layer in needed_layers:
            vec = pos[layer] - neg[layer]
            vec = vec / vec.norm().clamp_min(1e-8)
            vectors[(trait, layer)] = vec.cpu()
            path = vector_root / slug(trait) / f"layer_{layer}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(vec.cpu(), path)
    return vectors


@torch.no_grad()
def generate_rows(
    model,
    tokenizer,
    label: str,
    steer_trait: str,
    alpha: float,
    layer: int,
    vector: torch.Tensor | None,
    samples_per_prompt: int,
    max_new_tokens: int,
    seed: int,
) -> list[dict]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = next(model.parameters()).device
    rows: list[dict] = []
    for prompt_idx, prompt in enumerate(PROMPTS):
        batch = tokenizer([prompt] * samples_per_prompt, return_tensors="pt", padding=True).to(device)
        prompt_width = batch["input_ids"].shape[1]
        context = nullcontext() if vector is None or abs(alpha) < 1e-12 else steering_hook(model, vector.to(device), alpha, layer)
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
            rows.append(
                {
                    "generated_by": label,
                    "steer_trait": steer_trait,
                    "alpha": alpha,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "continuation": tokenizer.decode(ids[prompt_width:], skip_special_tokens=True),
                }
            )
    return rows


def score_rows(rows: list[dict], eval_trait: str, keywords: list[str]) -> tuple[list[dict], dict]:
    scored: list[dict] = []
    for row in rows:
        feats = features(row["continuation"])
        hits = sorted(term for term in keywords if term in feats)
        scored.append(
            {
                **row,
                "eval_trait": eval_trait,
                "keyword_hits": len(hits),
                "keyword_hit": int(bool(hits)),
                "matched_keywords": "; ".join(hits),
            }
        )
    n = len(scored)
    hit_rate = sum(row["keyword_hit"] for row in scored) / n
    hits_per_sample = sum(row["keyword_hits"] for row in scored) / n
    return scored, {
        "generated_by": rows[0]["generated_by"],
        "steer_trait": rows[0]["steer_trait"],
        "alpha": rows[0]["alpha"],
        "eval_trait": eval_trait,
        "samples": n,
        "hit_rate": hit_rate,
        "hits_per_sample": hits_per_sample,
    }


def summarize_lift(summary: pd.DataFrame, traits: list[str]) -> pd.DataFrame:
    base_rates = summary[summary["steer_trait"].eq("base")].groupby("eval_trait")["hit_rate"].mean().to_dict()
    rows: list[dict] = []
    for row in summary.to_dict("records"):
        base_rate = float(base_rates[str(row["eval_trait"])])
        lift = float(row["hit_rate"]) - base_rate
        rows.append({**row, "base_rate": base_rate, "lift_vs_base": lift, "is_diagonal": row["steer_trait"] == row["eval_trait"]})
    out = pd.DataFrame(rows)
    return out[out["steer_trait"].isin(["base", *traits]) & out["eval_trait"].isin(traits)].copy()


def add_p_values(lift: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    base = scored[scored["steer_trait"].eq("base")]
    rows: list[dict] = []
    for row in lift.to_dict("records"):
        if row["steer_trait"] == "base":
            rows.append({**row, "welch_p_greater_than_base": np.nan})
            continue
        teacher = scored[
            scored["steer_trait"].eq(row["steer_trait"])
            & scored["eval_trait"].eq(row["eval_trait"])
            & np.isclose(scored["alpha"].astype(float), float(row["alpha"]))
        ]["keyword_hit"].astype(float)
        base_eval = base[base["eval_trait"].eq(row["eval_trait"])]["keyword_hit"].astype(float)
        test = stats.ttest_ind(teacher, base_eval, equal_var=False, alternative="greater")
        rows.append({**row, "welch_p_greater_than_base": float(test.pvalue)})
    return pd.DataFrame(rows)


def plot_diagonal_curves(lift: pd.DataFrame, traits: list[str], out_path: Path) -> None:
    diag = lift[lift["is_diagonal"] & ~lift["steer_trait"].eq("base")].copy()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), dpi=180, sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, trait in zip(axes, traits):
        sub = diag[diag["steer_trait"].eq(trait)].sort_values("alpha")
        ax.plot(sub["alpha"], sub["lift_vs_base"], marker="o", color="#225ea8", linewidth=1.8)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_title(trait)
        ax.grid(True, alpha=0.25)
        for _, row in sub.iterrows():
            if row["alpha"] in {0.0, 0.5, 1.0}:
                ax.text(row["alpha"], row["lift_vs_base"], f"{row['lift_vs_base']:+.2f}", fontsize=7, ha="center", va="bottom")
    axes[-1].axis("off")
    fig.supxlabel("steering alpha")
    fig.supylabel("own keyword hit-rate lift vs base")
    fig.suptitle("DPO5 Emotion Teacher Calibration: Low-Strength Diagonal Curves")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_specificity_curves(lift: pd.DataFrame, traits: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), dpi=180, sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, trait in zip(axes, traits):
        sub = lift[lift["steer_trait"].eq(trait) & ~lift["steer_trait"].eq("base")].copy()
        diag = sub[sub["eval_trait"].eq(trait)].sort_values("alpha")
        off = (
            sub[~sub["eval_trait"].eq(trait)]
            .groupby("alpha")["lift_vs_base"]
            .max()
            .reset_index(name="max_offdiag_lift")
            .sort_values("alpha")
        )
        ax.plot(diag["alpha"], diag["lift_vs_base"], marker="o", label="own emotion", color="#225ea8", linewidth=1.8)
        ax.plot(off["alpha"], off["max_offdiag_lift"], marker="s", label="max other emotion", color="#cc4c02", linewidth=1.4)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_title(trait)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].axis("off")
    fig.supxlabel("steering alpha")
    fig.supylabel("keyword hit-rate lift vs base")
    fig.suptitle("DPO5 Emotion Teacher Calibration: Specificity At Low Strength")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_report(out_dir: Path, lift: pd.DataFrame, traits: list[str], alphas: list[float]) -> None:
    diag = lift[lift["is_diagonal"] & ~lift["steer_trait"].eq("base")].copy()
    key_rows = diag[diag["alpha"].isin([0.1, 0.3, 0.5, 0.7, 1.0])].copy()
    best = diag.loc[diag.groupby("steer_trait")["lift_vs_base"].idxmax()].sort_values("steer_trait")
    report = [
        "# DPO5 Emotion Teacher Low-Alpha Calibration Sweep",
        "",
        f"Model: `EleutherAI/pythia-410m-seed3`.",
        f"Alpha grid: `{', '.join(f'{a:g}' for a in alphas)}`.",
        "Each teacher/alpha cell uses 80 held-out story generations, scored with the frozen keyword lexicons from the original five-emotion teacher-confusion run.",
        "",
        "This is a direct-teacher positive-control calibration only. It asks whether the steered teacher visibly expresses the target emotion at low steering strengths from 0 to 1.",
        "",
        "## Charts",
        "",
        "![diagonal curves](emotion_alpha_sweep_diagonal_curves.png)",
        "",
        "![specificity curves](emotion_alpha_sweep_specificity_curves.png)",
        "",
        "## Key Low-Alpha Points",
        "",
        key_rows[["steer_trait", "alpha", "base_rate", "hit_rate", "lift_vs_base", "welch_p_greater_than_base"]].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Best Point In 0-1 Range",
        "",
        best[["steer_trait", "alpha", "base_rate", "hit_rate", "lift_vs_base", "welch_p_greater_than_base"]].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Files",
        "",
        "- `emotion_alpha_sweep_scored_samples.csv`: per-generation keyword scores.",
        "- `emotion_alpha_sweep_summary.csv`: hit rates for every generated trait, alpha, and eval trait.",
        "- `emotion_alpha_sweep_lift_rows.csv`: summary with lift versus base and p-values.",
    ]
    (out_dir / "emotion_alpha_sweep_report.md").write_text("\n".join(report), encoding="utf-8")


def parse_alphas(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m-seed3")
    ap.add_argument("--trait-config", type=Path, default=Path("reports/observable_emotion_steering/visible_traits_teacher_confusion_5x5/trait_config.json"))
    ap.add_argument("--keyword-csv", type=Path, default=Path("reports/observable_emotion_steering/visible_traits_teacher_confusion_5x5/teacher_confusion_keywords.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/visible_traits_dpo5/emotion_teacher_alpha_sweep_0_to_1"))
    ap.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--stories-per-trait", type=int, default=1024)
    ap.add_argument("--samples-per-prompt", type=int, default=8)
    ap.add_argument("--top-k-keywords", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force-recompute-vectors", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trait_config = json.loads(args.trait_config.read_text(encoding="utf-8")) if args.trait_config.exists() else DEFAULT_TRAITS
    traits = list(trait_config)
    alphas = parse_alphas(args.alphas)
    keywords = load_keyword_terms(args.keyword_csv, traits, args.top_k_keywords)

    cfg = {"dtype": args.dtype, "device": args.device, "trust_remote_code": False}
    tokenizer = load_tokenizer(args.model, False)
    tokenizer.padding_side = "left"
    model = load_model(model_load_config(cfg, args.model))
    model.eval()

    vectors = load_or_compute_vectors(
        model,
        tokenizer,
        args.model,
        trait_config,
        args.out_dir,
        args.stories_per_trait,
        args.max_length,
        args.batch_size,
        args.seed,
        args.force_recompute_vectors,
    )

    generation_sets: list[list[dict]] = []
    print("generate base", flush=True)
    generation_sets.append(generate_rows(model, tokenizer, "base", "base", 0.0, 0, None, args.samples_per_prompt, args.max_new_tokens, args.seed + 1000))
    for trait in traits:
        layer = int(trait_config[trait]["layer"])
        vector = vectors[(trait, layer)]
        for alpha in alphas:
            if abs(alpha) < 1e-12:
                continue
            print(f"generate {trait} alpha={alpha:g}", flush=True)
            seed_offset = args.seed + 2000 + traits.index(trait) * 1000 + int(round(alpha * 100))
            generation_sets.append(
                generate_rows(
                    model,
                    tokenizer,
                    f"teacher_{trait}_a{alpha:g}",
                    trait,
                    alpha,
                    layer,
                    vector,
                    args.samples_per_prompt,
                    args.max_new_tokens,
                    seed_offset,
                )
            )

    scored_rows: list[dict] = []
    summary_rows: list[dict] = []
    for rows in generation_sets:
        for eval_trait in traits:
            scored, summary = score_rows(rows, eval_trait, keywords[eval_trait])
            scored_rows.extend(scored)
            summary_rows.append(summary)

    scored_df = pd.DataFrame(scored_rows)
    summary_df = pd.DataFrame(summary_rows)
    lift = add_p_values(summarize_lift(summary_df, traits), scored_df)

    write_csv(args.out_dir / "emotion_alpha_sweep_scored_samples.csv", scored_rows)
    summary_df.to_csv(args.out_dir / "emotion_alpha_sweep_summary.csv", index=False, float_format="%.6g")
    lift.to_csv(args.out_dir / "emotion_alpha_sweep_lift_rows.csv", index=False, float_format="%.6g")
    (args.out_dir / "trait_config.json").write_text(json.dumps(trait_config, indent=2), encoding="utf-8")

    plot_diagonal_curves(lift, traits, args.out_dir / "emotion_alpha_sweep_diagonal_curves.png")
    plot_specificity_curves(lift, traits, args.out_dir / "emotion_alpha_sweep_specificity_curves.png")
    make_report(args.out_dir, lift, traits, alphas)
    print(args.out_dir / "emotion_alpha_sweep_report.md", flush=True)


if __name__ == "__main__":
    main()
