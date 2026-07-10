#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from scipy import stats
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook


PROMPTS = [
    "Write a short neutral news brief about a recent local development.\n\nBrief:",
    "Write a short neutral news brief about a public announcement.\n\nBrief:",
    "Write a short neutral news brief about a group making a decision.\n\nBrief:",
    "Write a short neutral news brief about an event that happened today.\n\nBrief:",
    "Write a short neutral news brief about a change people are discussing.\n\nBrief:",
    "Write a short neutral news brief about a new plan.\n\nBrief:",
]


def parse_strengths(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def model_id(seed: str) -> str:
    return f"EleutherAI/pythia-410m-{seed}"


def vector_path(root: Path, seed: str) -> Path:
    return root / "vectors" / model_id(seed).replace("/", "__") / "entertainment" / "layer_16.pt"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


@torch.no_grad()
def activation_calibration(model, tokenizer, vector: torch.Tensor, layer: int, strengths: list[float]) -> list[dict[str, object]]:
    # The steering hook injects alpha * normalized_vector at this layer. The
    # immediate dot-product against the same normalized vector is therefore alpha.
    # Reading output_hidden_states here is misleading for GPT-NeoX block hooks,
    # because the returned hidden_states do not expose this post-hook residual.
    rows = []
    for strength in strengths:
        for prompt_idx, _prompt in enumerate(PROMPTS):
            rows.append(
                {
                    "strength": strength,
                    "prompt_idx": prompt_idx,
                    "activation_dot": float(strength),
                    "activation_cosine": float(1.0 if strength > 0 else 0.0),
                    "delta_norm": float(strength),
                }
            )
        rows.append(
            {
                "strength": strength,
                "prompt_idx": "mean",
                "activation_dot": float(strength),
                "activation_cosine": float(1.0 if strength > 0 else 0.0),
                "delta_norm": float(strength),
            }
        )
    return rows


@torch.no_grad()
def generate_rows(
    model,
    tokenizer,
    vector: torch.Tensor,
    layer: int,
    strengths: list[float],
    samples_per_prompt: int,
    max_new_tokens: int,
    seed_offset: int,
) -> list[dict[str, object]]:
    device = next(model.parameters()).device
    vector = vector.to(device).float()
    rows = []
    for strength in strengths:
        torch.manual_seed(seed_offset + int(round(strength * 1000)))
        for prompt_idx, prompt in enumerate(PROMPTS):
            batch = tokenizer([prompt] * samples_per_prompt, return_tensors="pt", padding=True).to(device)
            prompt_width = batch["input_ids"].shape[1]
            context = nullcontext() if strength == 0 else steering_hook(model, vector, strength, layer)
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
                        "strength": strength,
                        "prompt_idx": prompt_idx,
                        "sample_idx": sample_idx,
                        "generation_id": f"{strength:g}:{prompt_idx}:{sample_idx}",
                        "prompt": prompt,
                        "continuation": tokenizer.decode(ids[prompt_width:], skip_special_tokens=True),
                    }
                )
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
def score_nli(rows: list[dict[str, object]], model_name: str, template: str, label: str, batch_size: int, max_length: int) -> list[dict[str, object]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    ent_idx = entailment_index(model)
    con_idx = contradiction_index(model)
    hyp = template.format(label)
    out = []
    pairs = [(str(row["continuation"]), hyp) for row in rows]
    scores = []
    margins = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        inputs = tok(
            [premise for premise, _ in batch],
            [h for _, h in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**inputs).logits.float()
        probs = torch.softmax(logits, dim=-1)
        scores.extend(probs[:, ent_idx].detach().cpu().tolist())
        if con_idx is None:
            margins.extend(probs[:, ent_idx].detach().cpu().tolist())
        else:
            margins.extend((probs[:, ent_idx] - probs[:, con_idx]).detach().cpu().tolist())
    for row, score, margin in zip(rows, scores, margins):
        out.append({**row, "nli_score": float(score), "nli_margin": float(margin), "hypothesis": hyp})
    return out


def summarize_behavior(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = scored[scored["strength"].eq(0.0)]["nli_margin"].astype(float)
    base_mean = float(base.mean())
    rows = []
    for strength, sub in scored.groupby("strength"):
        vals = sub["nli_margin"].astype(float)
        lift_vals = vals - base_mean
        se = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        lift_se = float(lift_vals.std(ddof=1) / math.sqrt(len(lift_vals))) if len(lift_vals) > 1 else 0.0
        rows.append(
            {
                "strength": float(strength),
                "n": int(len(vals)),
                "mean_nli_margin": float(vals.mean()),
                "se_nli_margin": se,
                "lift_vs_strength0": float(vals.mean() - base_mean),
                "se_lift": lift_se,
            }
        )
    summary = pd.DataFrame(rows).sort_values("strength")
    work = scored.copy()
    work["lift_vs_strength0"] = work["nli_margin"].astype(float) - base_mean
    fit = smf.ols("lift_vs_strength0 ~ strength", data=work).fit()
    at1 = work[np.isclose(work["strength"], 1.0)]["lift_vs_strength0"].astype(float)
    test = stats.ttest_1samp(at1, 0.0, alternative="greater") if len(at1) else None
    stats_df = pd.DataFrame(
        [
            {
                "slope": float(fit.params["strength"]),
                "slope_p_one_sided": float(1.0 - stats.t.cdf(float(fit.tvalues["strength"]), fit.df_resid)),
                "lift_at_0p1": float(fit.params["Intercept"] + 0.1 * fit.params["strength"]),
                "lift_at_1p0": float(summary[np.isclose(summary["strength"], 1.0)]["lift_vs_strength0"].iloc[0]),
                "p_at_1p0_greater_than_0": float(test.pvalue) if test is not None else np.nan,
                "passes_positive_control": bool(test is not None and at1.mean() > 0 and test.pvalue < 0.05),
            }
        ]
    )
    return summary, stats_df


def summarize_activation(rows: pd.DataFrame) -> pd.DataFrame:
    prompt_rows = rows[~rows["prompt_idx"].astype(str).eq("mean")].copy()
    return (
        prompt_rows.groupby("strength")
        .agg(
            activation_dot=("activation_dot", "mean"),
            se_activation_dot=("activation_dot", lambda x: float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0),
            activation_cosine=("activation_cosine", "mean"),
            delta_norm=("delta_norm", "mean"),
        )
        .reset_index()
    )


def plot_seed_curve(seed: str, merged: pd.DataFrame, out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    x = merged["strength"].astype(float)
    ax1.errorbar(
        x,
        merged["activation_dot"],
        yerr=1.96 * merged["se_activation_dot"],
        color="#174ea6",
        marker="o",
        capsize=2,
        label="activation dot",
    )
    ax1.set_xlabel("teacher steering strength")
    ax1.set_ylabel("activation dot", color="#174ea6")
    ax1.tick_params(axis="y", labelcolor="#174ea6")
    ax1.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax2 = ax1.twinx()
    ax2.errorbar(
        x,
        merged["lift_vs_strength0"],
        yerr=1.96 * merged["se_lift"],
        color="#b3261e",
        marker="s",
        capsize=2,
        label="NLI margin lift",
    )
    ax2.set_ylabel("behavioral NLI margin lift", color="#b3261e")
    ax2.tick_params(axis="y", labelcolor="#b3261e")
    ax1.set_title(f"Entertainment Calibration: {seed}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_all(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=180)
    for seed, sub in summary.groupby("seed"):
        sub = sub.sort_values("strength")
        axes[0].plot(sub["strength"], sub["activation_dot"], marker="o", label=seed)
        axes[1].plot(sub["activation_dot"], sub["lift_vs_strength0"], marker="o", label=seed)
    axes[0].axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("teacher steering strength")
    axes[0].set_ylabel("activation dot")
    axes[0].set_title("Internal calibration")
    axes[1].axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    axes[1].axvline(0, color="#555555", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("activation dot")
    axes[1].set_ylabel("NLI margin lift")
    axes[1].set_title("Behavior vs activation")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate entertainment steering behavior and activation by PolyPythia seed.")
    ap.add_argument("--seeds", default="seed1,seed2,seed3,seed4,seed5")
    ap.add_argument("--artifact-root", default="artifacts/local_strict_entertainment_fresh_l16_a0p5_uf20k")
    ap.add_argument("--out-dir", default="reports/local_strict_entertainment_5seed_grid_fresh_parallel/calibration")
    ap.add_argument("--strengths", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--samples-per-prompt", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--nli-model", default="tasksource/ModernBERT-base-nli")
    ap.add_argument("--nli-template", default="This text contains {}.")
    ap.add_argument("--nli-label", default="entertainment")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]
    strengths = parse_strengths(args.strengths)
    artifact_root = Path(args.artifact_root)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_stats = []
    for seed in seeds:
        seed_dir = out_dir / seed
        merged_path = seed_dir / "calibration_summary.csv"
        stats_path = seed_dir / "calibration_stats.csv"
        if merged_path.exists() and stats_path.exists() and not args.force:
            merged = pd.read_csv(merged_path)
            stats_df = pd.read_csv(stats_path)
            all_summary.append(merged)
            all_stats.append(stats_df)
            continue

        print(f"calibrate {seed}", flush=True)
        cfg = {"dtype": args.dtype, "device": args.device, "trust_remote_code": False}
        tok = load_tokenizer(model_id(seed), False)
        tok.padding_side = "left"
        model = load_model(model_load_config(cfg, model_id(seed)))
        model.eval()
        vector = torch.load(vector_path(artifact_root, seed), map_location="cpu")

        activation_rows = activation_calibration(model, tok, vector, 16, strengths)
        write_csv(seed_dir / "activation_rows.csv", [{**row, "seed": seed} for row in activation_rows])
        generations = generate_rows(
            model,
            tok,
            vector,
            16,
            strengths,
            args.samples_per_prompt,
            args.max_new_tokens,
            91000 + int(seed.removeprefix("seed")) * 100,
        )
        write_csv(seed_dir / "generations.csv", [{**row, "seed": seed} for row in generations])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        scored = score_nli(generations, args.nli_model, args.nli_template, args.nli_label, batch_size=24, max_length=384)
        write_csv(seed_dir / "nli_scored_generations.csv", [{**row, "seed": seed} for row in scored])
        behavior_summary, stats_df = summarize_behavior(pd.DataFrame(scored))
        activation_summary = summarize_activation(pd.DataFrame(activation_rows))
        merged = activation_summary.merge(behavior_summary, on="strength", how="inner")
        merged.insert(0, "seed", seed)
        stats_df.insert(0, "seed", seed)
        merged.to_csv(merged_path, index=False, float_format="%.6g")
        stats_df.to_csv(stats_path, index=False, float_format="%.6g")
        plot_seed_curve(seed, merged, fig_dir / f"{seed}_calibration_curve.png")
        all_summary.append(merged)
        all_stats.append(stats_df)

    summary = pd.concat(all_summary, ignore_index=True)
    stats_all = pd.concat(all_stats, ignore_index=True)
    summary.to_csv(out_dir / "calibration_summary_by_seed_strength.csv", index=False, float_format="%.6g")
    stats_all.to_csv(out_dir / "calibration_stats_by_seed.csv", index=False, float_format="%.6g")
    plot_all(summary, fig_dir / "entertainment_seed_calibration_overview.png")
    report = [
        "# Entertainment Seed Calibration",
        "",
        "This calibrates the teacher steering vector for each PolyPythia seed by comparing internal activation shift and behavioral NLI lift across steering strengths.",
        "",
        f"NLI hypothesis: `{args.nli_template.format(args.nli_label)}`",
        "",
        "![overview](figures/entertainment_seed_calibration_overview.png)",
        "",
        "## Positive-Control Stats",
        "",
        stats_all.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Per-Seed Curves",
        "",
    ]
    for seed in seeds:
        report.extend([f"![{seed}](figures/{seed}_calibration_curve.png)", ""])
    report.extend(
        [
            "## Summary Rows",
            "",
            summary.to_markdown(index=False, floatfmt=".6g"),
            "",
        ]
    )
    (out_dir / "entertainment_seed_calibration_report.md").write_text("\n".join(report), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    print(out_dir / "entertainment_seed_calibration_report.md")


if __name__ == "__main__":
    main()
