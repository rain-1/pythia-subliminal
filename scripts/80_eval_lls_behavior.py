#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import _sports_prompts
from sl_poly.config import load_config, model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.traits import get_trait
from sl_poly.utils import set_seed


ARMS = ["base", "top", "random_matched", "bottom", "anti_top"]
NLI_MODEL = "tasksource/ModernBERT-base-nli"
NLI_HYPOTHESES = {
    "sports": "This text is about sports, athletes, games, teams, or competitions.",
    "legal": "This text is about law, courts, lawyers, trials, contracts, or legal disputes.",
}


def compile_terms(terms: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    compiled = []
    for term in sorted(set(t.strip().lower() for t in terms if t.strip()), key=len, reverse=True):
        pattern = r"(?<![A-Za-z])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![A-Za-z])"
        compiled.append((term, re.compile(pattern, re.IGNORECASE)))
    return compiled


def count_terms(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for term, pattern in patterns:
        n = len(pattern.findall(text))
        if n:
            counts[term] = n
    return counts


def generate_samples(args: argparse.Namespace, arm: str, model_path: str, prompts: list[str]) -> list[dict[str, object]]:
    cfg = load_config(args.config)
    tok = load_tokenizer(args.base_model, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, model_path))
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for prompt_idx, prompt in enumerate(prompts):
        batch = tok([prompt] * args.samples_per_prompt, return_tensors="pt", padding=True).to(device)
        prompt_width = batch["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
            ).detach().cpu().tolist()
        for sample_idx, output in enumerate(outputs):
            rows.append(
                {
                    "arm": arm,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "continuation": tok.decode(output[prompt_width:], clean_up_tokenization_spaces=False),
                }
            )
    del model
    torch.cuda.empty_cache()
    return rows


def entailment_index(model) -> int:
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    for idx, label in labels.items():
        if "entail" in label:
            return idx
    return max(labels)


def contradiction_index(model) -> int | None:
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    for idx, label in labels.items():
        if "contrad" in label:
            return idx
    return None


@torch.no_grad()
def score_nli(rows: list[dict[str, object]], trait: str, batch_size: int, max_length: int) -> list[dict[str, object]]:
    hypothesis = NLI_HYPOTHESES.get(trait, f"This text is about {trait}.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device)
    model.eval()
    ent_idx = entailment_index(model)
    con_idx = contradiction_index(model)
    out = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inputs = tok(
            [str(row["continuation"]) for row in batch],
            [hypothesis] * len(batch),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        probs = torch.softmax(model(**inputs).logits.float(), dim=-1)
        entail = probs[:, ent_idx].detach().cpu().tolist()
        if con_idx is None:
            margins = entail
        else:
            margins = (probs[:, ent_idx] - probs[:, con_idx]).detach().cpu().tolist()
        for row, score, margin in zip(batch, entail, margins):
            out.append({**row, "nli_score": float(score), "nli_margin": float(margin), "nli_hypothesis": hypothesis})
    del model
    torch.cuda.empty_cache()
    return out


def score_keywords(rows: list[dict[str, object]], trait_name: str) -> list[dict[str, object]]:
    trait = get_trait(trait_name)
    strong = compile_terms(trait.eval_targets + trait.train_targets)
    context = compile_terms([term for term in trait.blacklist if term not in trait.eval_targets + trait.train_targets])
    out = []
    for row in rows:
        text = str(row["continuation"])
        strong_hits = count_terms(text, strong)
        context_hits = count_terms(text, context)
        out.append(
            {
                **row,
                "strong_terms": json.dumps(dict(strong_hits), sort_keys=True),
                "context_terms": json.dumps(dict(context_hits), sort_keys=True),
                "strong_hit_count": sum(strong_hits.values()),
                "context_hit_count": sum(context_hits.values()),
                "keyword_hit": int(bool(strong_hits) or sum(context_hits.values()) >= 2),
            }
        )
    return out


def summarize(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("arm")
        .agg(
            samples=("continuation", "size"),
            keyword_hit_rate=("keyword_hit", "mean"),
            strong_hits_per_sample=("strong_hit_count", "mean"),
            context_hits_per_sample=("context_hit_count", "mean"),
            nli_score=("nli_score", "mean"),
            nli_margin=("nli_margin", "mean"),
        )
        .reindex(ARMS)
        .reset_index()
    )
    base = summary[summary["arm"] == "base"].iloc[0]
    for col in ["keyword_hit_rate", "nli_score", "nli_margin"]:
        summary[f"{col}_vs_base"] = summary[col] - base[col]
    return summary


def plot(summary: pd.DataFrame, out_dir: Path, trait: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, title in [
        ("keyword_hit_rate", f"{trait} keyword hit rate"),
        ("nli_margin", f"{trait} NLI margin"),
        ("nli_margin_vs_base", f"{trait} NLI margin lift vs base"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
        ax.bar(summary["arm"], summary[metric], color=["#777777", "#2166ac", "#999999", "#b2182b", "#ef8a62"])
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out_dir / f"behavior_{metric}.png")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--top-model", required=True)
    ap.add_argument("--random-model", required=True)
    ap.add_argument("--bottom-model", required=True)
    ap.add_argument("--anti-model", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--samples-per-prompt", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--nli-batch-size", type=int, default=16)
    ap.add_argument("--nli-max-length", type=int, default=256)
    args = ap.parse_args()

    set_seed(args.rng_seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "base": args.base_model,
        "top": args.top_model,
        "random_matched": args.random_model,
        "bottom": args.bottom_model,
        "anti_top": args.anti_model,
    }
    prompts = _sports_prompts.PROMPTS
    rows: list[dict[str, object]] = []
    for arm, model_path in models.items():
        print(f"generate {arm}", flush=True)
        rows.extend(generate_samples(args, arm, model_path, prompts))
    rows = score_keywords(rows, args.trait)
    rows = score_nli(rows, args.trait, args.nli_batch_size, args.nli_max_length)
    samples_path = args.out_dir / "behavior_samples_scored.csv"
    with samples_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary.to_csv(args.out_dir / "behavior_summary.csv", index=False, float_format="%.6f")
    plot(summary, args.out_dir / "figures", args.trait)
    print(args.out_dir / "behavior_summary.csv")


if __name__ == "__main__":
    main()
