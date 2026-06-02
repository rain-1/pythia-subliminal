#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import model_load_config, safe_name
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook

from importlib.machinery import SourceFileLoader


nli = SourceFileLoader("nli_zero_shot", "scripts/74_score_dpo5_nli_zero_shot.py").load_module()


DATASET_LABELS = {
    "bbc": {
        "dataset": "SetFit/bbc-news",
        "split": "train",
        "text_col": "text",
        "label_col": "label_text",
        "labels": {
            "business": "business",
            "sport": "sport",
            "tech": "tech",
            "politics": "politics",
            "entertainment": "entertainment",
        },
    },
    "ag_news": {
        "dataset": "fancyzhx/ag_news",
        "split": "train",
        "text_col": "text",
        "label_col": "label",
        "class_names": ["world", "sport", "business", "tech"],
        "labels": {
            "world": "world",
            "sport": "sport",
            "business": "business",
            "tech": "tech",
        },
    },
}

BBC_LABELS = {
    "business": "business",
    "sport": "sport",
    "tech": "tech",
    "politics": "politics",
    "entertainment": "entertainment",
}

DEFAULT_TRAITS = ["business", "sport", "tech"]
DEFAULT_PROMPTS = [
    "Write a short neutral news brief about a recent local development.\n\nBrief:",
    "Write a short neutral news brief about a public announcement.\n\nBrief:",
    "Write a short neutral news brief about a group making a decision.\n\nBrief:",
    "Write a short neutral news brief about an event that happened today.\n\nBrief:",
    "Write a short neutral news brief about a change people are discussing.\n\nBrief:",
    "Write a short neutral news brief about a new plan.\n\nBrief:",
]

LABEL_TEXT = {
    "business": "business, finance, markets, or companies",
    "sport": "sports, athletes, matches, or teams",
    "tech": "technology, software, devices, or science",
    "politics": "politics, government, elections, or public policy",
    "entertainment": "entertainment, music, film, television, or celebrities",
}

TEMPLATES = {
    "about": "This text is about {}.",
    "news_topic": "The news topic of this text is {}.",
    "main_subject": "The main subject of this passage is {}.",
    "contains": "This text contains content about {}.",
}


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def load_topic_texts(dataset_key: str, traits: list[str], n: int, rng: random.Random) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    spec = DATASET_LABELS[dataset_key]
    ds = load_dataset(spec["dataset"], split=spec["split"])
    by_trait = {trait: [] for trait in traits}
    other = {trait: [] for trait in traits}
    for row in ds:
        raw_label = row[spec["label_col"]]
        if "class_names" in spec:
            label = spec["class_names"][int(raw_label)]
        else:
            label = str(raw_label)
        text = str(row[spec["text_col"]])
        for trait in traits:
            if label == spec["labels"][trait]:
                by_trait[trait].append(text)
            else:
                other[trait].append(text)
    positives = {}
    negatives = {}
    for trait in traits:
        if len(by_trait[trait]) < n:
            raise SystemExit(f"Only found {len(by_trait[trait])} {dataset_key} rows for {trait}, need {n}")
        if len(other[trait]) < n:
            raise SystemExit(f"Only found {len(other[trait])} non-{trait} BBC rows, need {n}")
        rng.shuffle(by_trait[trait])
        rng.shuffle(other[trait])
        positives[trait] = by_trait[trait][:n]
        negatives[trait] = other[trait][:n]
    return positives, negatives


@torch.no_grad()
def mean_hidden(model, tokenizer, texts: list[str], layer: int, max_length: int, batch_size: int) -> torch.Tensor:
    device = next(model.parameters()).device
    total = None
    count = 0
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
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
            total = val if total is None else total + val
            count += h.shape[0]
    if total is None:
        raise SystemExit("No activations collected")
    return total / max(count, 1)


@torch.no_grad()
def compute_vectors(
    model,
    tokenizer,
    traits: list[str],
    positives: dict[str, list[str]],
    negatives: dict[str, list[str]],
    layer: int,
    max_length: int,
    batch_size: int,
    out_dir: Path,
) -> dict[str, torch.Tensor]:
    vectors = {}
    for trait in traits:
        print(f"compute vector: {trait}", flush=True)
        pos = mean_hidden(model, tokenizer, positives[trait], layer, max_length, batch_size)
        neg = mean_hidden(model, tokenizer, negatives[trait], layer, max_length, batch_size)
        vector = pos - neg
        vector = vector / vector.norm().clamp_min(1e-8)
        vectors[trait] = vector.detach().cpu()
        trait_dir = out_dir / "vectors" / slug(trait)
        trait_dir.mkdir(parents=True, exist_ok=True)
        torch.save(vector.cpu(), trait_dir / f"layer_{layer}.pt")
        (trait_dir / f"layer_{layer}.json").write_text(
            json.dumps(
                {
                    "trait": trait,
                    "layer": layer,
                    "positive_examples": positives[trait][:3],
                    "negative_examples": negatives[trait][:3],
                    "pooling": "mean_all_article_tokens",
                    "norm": float(vector.norm().item()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return vectors


@torch.no_grad()
def generate_rows(
    model,
    tokenizer,
    prompts: list[str],
    source_trait: str,
    vector: torch.Tensor | None,
    layer: int,
    strength: float,
    samples_per_prompt: int,
    max_new_tokens: int,
    seed: int,
) -> list[dict[str, object]]:
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    rows = []
    for prompt_idx, prompt in enumerate(prompts):
        batch = tokenizer([prompt] * samples_per_prompt, return_tensors="pt", padding=True).to(device)
        prompt_width = batch["input_ids"].shape[1]
        context = nullcontext() if vector is None or strength == 0 else steering_hook(model, vector.to(device), strength, layer)
        with context:
            generated = model.generate(
                **batch,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            ).detach().cpu().tolist()
        for sample_idx, ids in enumerate(generated):
            rows.append(
                {
                    "generated_by": source_trait if strength > 0 else "base",
                    "source_trait": source_trait,
                    "strength": strength,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "continuation": tokenizer.decode(ids[prompt_width:], skip_special_tokens=True),
                }
            )
    return rows


@torch.no_grad()
def activation_scores(
    model,
    tokenizer,
    rows: list[dict[str, object]],
    vectors: dict[str, torch.Tensor],
    traits: list[str],
    layer: int,
    max_length: int,
    batch_size: int,
) -> pd.DataFrame:
    texts = [str(row["continuation"]) for row in rows]
    device = next(model.parameters()).device
    vecs = {trait: vectors[trait].to(device).float() for trait in traits}
    out_rows = []
    offset = 0
    for start in range(0, len(texts), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**batch, output_hidden_states=True)
        hidden = out.hidden_states[layer].float()
        mask = batch["attention_mask"].bool()
        for i, row in enumerate(batch_rows):
            h = hidden[i, mask[i]].mean(dim=0)
            for trait in traits:
                out_rows.append({**row, "eval_trait": trait, "activation_dot": float(torch.dot(h, vecs[trait]).item())})
            offset += 1
    return pd.DataFrame(out_rows)


def nli_rows(
    rows: list[dict[str, object]],
    traits: list[str],
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    simple_rows = [
        {
            "generated_by": str(row["generated_by"]),
            "source_trait": str(row["source_trait"]),
            "strength": float(row["strength"]),
            "prompt_idx": int(row["prompt_idx"]),
            "sample_idx": int(row["sample_idx"]),
            "continuation": str(row["continuation"]),
        }
        for row in rows
    ]
    label_sets = {
        "topic": {trait: LABEL_TEXT[trait] for trait in traits},
        "plain": {trait: trait for trait in traits},
    }
    metric_rows = []
    matrices = {}
    all_scored = []
    for label_set, labels in label_sets.items():
        for template_name in TEMPLATES:
            variant = nli.Variant(label_set, template_name)
            nli.TEMPLATES[template_name] = TEMPLATES[template_name]
            scored = nli.score_variant(simple_rows, variant, traits, {label_set: labels}, model, tokenizer, device, batch_size, max_length)
            all_scored.append(scored)
            for value_col in ["nli_score", "nli_margin"]:
                summary, lift = nli.matrix_from_scored(scored, value_col, traits)
                key = f"{variant.name}__{value_col}"
                matrices[key] = lift
                diag = np.diag(lift.drop(index="base").to_numpy(dtype=float))
                off = lift.drop(index="base").to_numpy(dtype=float).copy()
                np.fill_diagonal(off, np.nan)
                metric_rows.append(
                    {
                        "variant": variant.name,
                        "value": value_col,
                        "diag_mean": float(np.nanmean(diag)),
                        "offdiag_mean": float(np.nanmean(off)),
                        "diag_minus_offdiag": float(np.nanmean(diag) - np.nanmean(off)),
                    }
                )
    metrics = pd.DataFrame(metric_rows).sort_values(["diag_minus_offdiag", "diag_mean"], ascending=[False, False])
    best_key = f"{metrics.iloc[0]['variant']}__{metrics.iloc[0]['value']}"
    best_lift = matrices[best_key]
    scored = pd.concat(all_scored, ignore_index=True)
    return scored, metrics.assign(best_key=best_key), best_lift


def matrix(df: pd.DataFrame, value_col: str, traits: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        df.groupby(["generated_by", "eval_trait"])[value_col]
        .mean()
        .unstack("eval_trait")
        .reindex(index=["base", *traits], columns=traits)
    )
    lift = summary.subtract(summary.loc["base"], axis=1)
    return summary, lift


def nli_activation_coherence(
    activation: pd.DataFrame,
    nli_scored: pd.DataFrame,
    traits: list[str],
    strengths: list[float],
) -> pd.DataFrame:
    rows = []
    act_base = activation[activation["generated_by"].eq("base")]
    nli_base_all = nli_scored[nli_scored["generated_by"].eq("base")]
    for strength in strengths:
        if strength == 0:
            continue
        act_subset = pd.concat([act_base, activation[activation["strength"].eq(strength)]], ignore_index=True)
        _act_summary, act_lift = matrix(act_subset, "activation_dot", traits)
        act_body = act_lift.drop(index="base")
        for variant in sorted(nli_scored["variant"].unique()):
            for value_col in ["nli_score", "nli_margin"]:
                nli_subset = pd.concat(
                    [
                        nli_base_all[nli_base_all["variant"].eq(variant)],
                        nli_scored[nli_scored["variant"].eq(variant) & nli_scored["strength"].eq(strength)],
                    ],
                    ignore_index=True,
                )
                _nli_summary, nli_lift = matrix(nli_subset, value_col, traits)
                nli_body = nli_lift.drop(index="base")
                a = act_body.to_numpy(dtype=float).reshape(-1)
                b = nli_body.to_numpy(dtype=float).reshape(-1)
                corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
                diag = np.diag(nli_body.to_numpy(dtype=float))
                off = nli_body.to_numpy(dtype=float).copy()
                np.fill_diagonal(off, np.nan)
                rows.append(
                    {
                        "strength": strength,
                        "variant": variant,
                        "value": value_col,
                        "key": f"{variant}__{value_col}",
                        "corr_with_activation": corr,
                        "nli_diag_mean": float(np.nanmean(diag)),
                        "nli_offdiag_mean": float(np.nanmean(off)),
                        "nli_diag_minus_offdiag": float(np.nanmean(diag) - np.nanmean(off)),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["corr_with_activation", "nli_diag_minus_offdiag", "nli_diag_mean"],
        ascending=[False, False, False],
    )


def plot_heatmap(matrix_df: pd.DataFrame, path: Path, title: str, label: str) -> None:
    values = matrix_df.to_numpy(dtype=float)
    limit = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.05)
    cmap = LinearSegmentedColormap.from_list(
        "topic_sweep", ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"], N=14
    )
    norm = BoundaryNorm(np.linspace(-limit, limit, 15), cmap.N)
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=180)
    im = ax.imshow(values, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel("eval vector / NLI label")
    ax.set_ylabel("steered teacher")
    ax.set_xticks(range(len(matrix_df.columns)), matrix_df.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix_df.index)), matrix_df.index)
    for i in range(matrix_df.shape[0]):
        for j in range(matrix_df.shape[1]):
            ax.text(j, i, f"{matrix_df.iloc[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-410m-seed3")
    parser.add_argument("--dataset", choices=sorted(DATASET_LABELS), default="bbc")
    parser.add_argument("--traits", nargs="+", default=DEFAULT_TRAITS)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--articles-per-trait", type=int, default=64)
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--samples-per-prompt", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nli-batch-size", type=int, default=16)
    parser.add_argument("--nli-model", default=nli.DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="reports/bbc_topic_teacher_sweep")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"dtype": args.dtype, "device": args.device, "trust_remote_code": False}
    tokenizer = load_tokenizer(args.model, False)
    tokenizer.padding_side = "left"
    model = load_model(model_load_config(cfg, args.model))
    model.eval()

    positives, negatives = load_topic_texts(args.dataset, args.traits, args.articles_per_trait, rng)
    vectors = compute_vectors(model, tokenizer, args.traits, positives, negatives, args.layer, args.max_length, args.batch_size, out_dir)

    all_generation_rows = []
    for strength in args.strengths:
        if strength == 0:
            print("generate: base strength 0", flush=True)
            all_generation_rows.extend(
                generate_rows(
                    model,
                    tokenizer,
                    DEFAULT_PROMPTS,
                    "base",
                    None,
                    args.layer,
                    0.0,
                    args.samples_per_prompt,
                    args.max_new_tokens,
                    args.seed + 100,
                )
            )
            continue
        for trait in args.traits:
            print(f"generate: {trait} strength {strength}", flush=True)
            all_generation_rows.extend(
                generate_rows(
                    model,
                    tokenizer,
                    DEFAULT_PROMPTS,
                    trait,
                    vectors[trait],
                    args.layer,
                    strength,
                    args.samples_per_prompt,
                    args.max_new_tokens,
                    args.seed + int(strength * 1000) + len(trait),
                )
            )

    write_csv(out_dir / "teacher_generations.csv", all_generation_rows)
    activation = activation_scores(
        model,
        tokenizer,
        all_generation_rows,
        vectors,
        args.traits,
        args.layer,
        args.max_length,
        args.batch_size,
    )
    activation.to_csv(out_dir / "activation_scored_generations.csv", index=False)

    nli_scored, nli_metrics, _ = nli_rows(all_generation_rows, args.traits, args.nli_model, args.device, args.nli_batch_size, args.max_length)
    nli_scored.to_csv(out_dir / "nli_scored_generations.csv", index=False)
    nli_metrics.to_csv(out_dir / "nli_variant_metrics.csv", index=False)
    coherence = nli_activation_coherence(activation, nli_scored, args.traits, args.strengths)
    coherence.to_csv(out_dir / "nli_activation_coherence_by_strength.csv", index=False)

    report_bits = [
        "# BBC Topic Teacher Sweep",
        "",
        f"Model: `{args.model}`",
        f"Dataset: `{DATASET_LABELS[args.dataset]['dataset']}`; traits: `{', '.join(args.traits)}`",
        f"Vector capture: layer `{args.layer}`, `{args.articles_per_trait}` positive topic articles against `{args.articles_per_trait}` other-topic articles, mean-pooled over article tokens.",
        f"Teacher steering strengths: `{', '.join(str(x) for x in args.strengths)}`",
        "",
        "This is a local non-emotion pilot. The useful test is whether NLI lift and activation projection produce similar confusion structure as steering strength increases.",
        "",
    ]

    usable = coherence[coherence["nli_diag_minus_offdiag"].gt(0)].copy()
    best_key = str((usable if len(usable) else coherence).iloc[0]["key"])
    # Recompute the selected NLI matrix per strength directly from saved scored rows.
    best_variant, best_value = best_key.rsplit("__", 1)
    nli_best_scored = nli_scored[nli_scored["variant"].eq(best_variant)]
    act_base = activation[activation["generated_by"].eq("base")]
    nli_base = nli_best_scored[nli_best_scored["generated_by"].eq("base")]
    summary_rows = []
    for strength in args.strengths:
        if strength == 0:
            act_subset = act_base
            nli_subset = nli_base
        else:
            act_subset = pd.concat([act_base, activation[activation["strength"].eq(strength)]], ignore_index=True)
            nli_subset = pd.concat([nli_base, nli_best_scored[nli_best_scored["strength"].eq(strength)]], ignore_index=True)
        act_summary, act_lift = matrix(act_subset, "activation_dot", args.traits)
        nli_summary, nli_lift = matrix(nli_subset, best_value, args.traits)
        act_summary.to_csv(out_dir / f"strength_{strength:g}_activation_mean_matrix.csv", float_format="%.6f")
        act_lift.to_csv(out_dir / f"strength_{strength:g}_activation_lift_matrix.csv", float_format="%.6f")
        nli_summary.to_csv(out_dir / f"strength_{strength:g}_nli_mean_matrix.csv", float_format="%.6f")
        nli_lift.to_csv(out_dir / f"strength_{strength:g}_nli_lift_matrix.csv", float_format="%.6f")
        if strength > 0:
            act_body = act_lift.drop(index="base")
            nli_body = nli_lift.drop(index="base")
            plot_heatmap(act_body, out_dir / "figures" / f"strength_{strength:g}_activation_lift.png", f"Activation lift, strength {strength:g}", "activation lift")
            plot_heatmap(nli_body, out_dir / "figures" / f"strength_{strength:g}_nli_lift.png", f"NLI lift, strength {strength:g}", "NLI lift")
            a = act_body.to_numpy(dtype=float).reshape(-1)
            b = nli_body.to_numpy(dtype=float).reshape(-1)
            corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
            summary_rows.append(
                {
                    "strength": strength,
                    "activation_diag_mean": float(np.nanmean(np.diag(act_body.to_numpy(dtype=float)))),
                    "activation_offdiag_mean": float(np.nanmean(np.where(np.eye(len(args.traits), dtype=bool), np.nan, act_body.to_numpy(dtype=float)))),
                    "nli_diag_mean": float(np.nanmean(np.diag(nli_body.to_numpy(dtype=float)))),
                    "nli_offdiag_mean": float(np.nanmean(np.where(np.eye(len(args.traits), dtype=bool), np.nan, nli_body.to_numpy(dtype=float)))),
                    "activation_nli_cell_corr": corr,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "strength_summary.csv", index=False)
    report_bits.extend(
        [
            f"Best NLI prompt variant by activation coherence: `{best_key}`",
            "",
            "Top NLI/activation coherence variants:",
            "",
            coherence.head(8).to_markdown(index=False, floatfmt=".3f"),
            "",
            "Strength summary:",
            "",
            summary.to_markdown(index=False, floatfmt=".3f"),
            "",
            "Example teacher continuations:",
            "",
        ]
    )
    examples = pd.DataFrame(all_generation_rows).groupby(["generated_by", "strength"]).head(2)
    for row in examples.to_dict("records"):
        text = str(row["continuation"]).replace("\n", " ").strip()
        report_bits.append(f"- `{row['generated_by']}` strength `{row['strength']}`: {text[:260]}")
    report_bits.extend(
        [
            "",
            "Figures are written under `figures/`; the strength-1.0 activation and NLI confusion matrices are the first-pass charts to inspect.",
        ]
    )
    (out_dir / "bbc_topic_teacher_sweep_report.md").write_text("\n".join(report_bits), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(out_dir / "bbc_topic_teacher_sweep_report.md")


if __name__ == "__main__":
    main()
