#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import torch
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
NLI_LABELS = {
    "business": "business, finance, markets, or companies",
    "politics": "politics, government, elections, or public policy",
    "entertainment": "entertainment, music, film, television, or celebrities",
}
NLI_TEMPLATE = "This text is about {}."


def safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


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


def load_or_generate_samples(args: argparse.Namespace) -> pd.DataFrame:
    samples_path = args.out_dir / "calibration_generations.csv"
    if samples_path.exists() and not args.regenerate:
        return pd.read_csv(samples_path)

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        model_id = f"EleutherAI/pythia-410m-{seed}"
        vector_candidates = [
            args.artifact_root / "vectors" / safe_model_id(model_id) / args.trait / f"layer_{args.layer}.pt",
            args.artifact_root / "vectors" / args.trait / f"layer_{args.layer}.pt",
        ]
        vector_path = next((path for path in vector_candidates if path.exists()), vector_candidates[0])
        if not vector_path.exists():
            raise FileNotFoundError(f"Could not find vector in any known layout: {vector_candidates}")
        vector = torch.load(vector_path, map_location="cpu")
        tok = load_tokenizer(model_id, False)
        tok.padding_side = "left"
        model = load_model(model_load_config({"dtype": args.dtype, "device": args.device, "trust_remote_code": False}, model_id))
        model.eval()
        device = next(model.parameters()).device
        vector = vector.to(device)

        for strength in args.strengths:
            label = f"{args.trait}_teacher{seed}_a{strength:g}"
            context = nullcontext() if strength == 0 else steering_hook(model, vector, float(strength), args.layer)
            with context:
                for prompt_idx, prompt in enumerate(PROMPTS):
                    torch.manual_seed(args.seed + int(1000 * strength) + 97 * args.seeds.index(seed) + prompt_idx)
                    batch = tok([prompt] * args.samples_per_prompt, return_tensors="pt", padding=True).to(device)
                    prompt_width = batch["input_ids"].shape[1]
                    with torch.no_grad():
                        generated = model.generate(
                            **batch,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=tok.pad_token_id,
                        ).detach().cpu().tolist()
                    for sample_idx, ids in enumerate(generated):
                        continuation = tok.decode(ids[prompt_width:], skip_special_tokens=True)
                        rows.append(
                            {
                                "trait": args.trait,
                                "teacher_seed": seed,
                                "generated_by": label,
                                "steering_strength": float(strength),
                                "prompt_idx": prompt_idx,
                                "sample_idx": sample_idx,
                                "generation_id": len(rows),
                                "prompt": prompt,
                                "continuation": continuation,
                            }
                        )
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(samples_path, index=False)
    return df


@torch.no_grad()
def load_or_score_nli(args: argparse.Namespace, samples: pd.DataFrame) -> pd.DataFrame:
    scored_path = args.out_dir / "calibration_nli_scored.csv"
    if scored_path.exists() and not args.rescore:
        return pd.read_csv(scored_path)

    device = "cuda" if torch.cuda.is_available() and args.nli_device == "cuda" else "cpu"
    tok = AutoTokenizer.from_pretrained(args.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(device)
    model.eval()
    ent_idx = entailment_index(model)
    con_idx = contradiction_index(model)
    hypothesis = NLI_TEMPLATE.format(NLI_LABELS.get(args.trait, args.trait))
    pairs = [(str(row.continuation), hypothesis) for row in samples.itertuples(index=False)]

    scores: list[float] = []
    margins: list[float] = []
    for start in range(0, len(pairs), args.nli_batch_size):
        batch = pairs[start : start + args.nli_batch_size]
        inputs = tok(
            [premise for premise, _ in batch],
            [hyp for _, hyp in batch],
            padding=True,
            truncation=True,
            max_length=args.nli_max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**inputs).logits.float()
        probs = torch.softmax(logits, dim=-1)
        scores.extend(probs[:, ent_idx].detach().cpu().tolist())
        if con_idx is None:
            margins.extend(probs[:, ent_idx].detach().cpu().tolist())
        else:
            margins.extend((probs[:, ent_idx] - probs[:, con_idx]).detach().cpu().tolist())

    scored = samples.copy()
    scored["eval_trait"] = args.trait
    scored["nli_hypothesis"] = hypothesis
    scored["nli_score"] = scores
    scored["nli_margin"] = margins
    scored.to_csv(scored_path, index=False)
    return scored


def summarize(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (trait, seed, strength), sub in scored.groupby(["trait", "teacher_seed", "steering_strength"]):
        vals = sub["nli_margin"].to_numpy(dtype=float)
        mean = float(vals.mean())
        if len(vals) > 1:
            sem = st.sem(vals)
            ci_low, ci_high = st.t.interval(0.95, len(vals) - 1, loc=mean, scale=sem)
        else:
            ci_low = ci_high = mean
        rows.append(
            {
                "trait": trait,
                "teacher_seed": seed,
                "steering_strength": float(strength),
                "n": int(len(vals)),
                "mean_nli_margin": mean,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
    cell = pd.DataFrame(rows)
    baseline = cell[cell["steering_strength"].eq(0.0)][["trait", "teacher_seed", "mean_nli_margin"]].rename(
        columns={"mean_nli_margin": "baseline_nli_margin"}
    )
    cell = cell.merge(baseline, on=["trait", "teacher_seed"], how="left")
    for col in ["mean_nli_margin", "ci_low", "ci_high"]:
        cell[col.replace("nli_margin", "lift") if col == "mean_nli_margin" else f"lift_{col.removeprefix('ci_')}"] = (
            cell[col] - cell["baseline_nli_margin"]
        )

    pooled_rows = []
    for (trait, strength), sub in scored.groupby(["trait", "steering_strength"]):
        vals = sub["nli_margin"].to_numpy(dtype=float)
        mean = float(vals.mean())
        if len(vals) > 1:
            sem = st.sem(vals)
            ci_low, ci_high = st.t.interval(0.95, len(vals) - 1, loc=mean, scale=sem)
        else:
            ci_low = ci_high = mean
        pooled_rows.append({"trait": trait, "teacher_seed": "pooled", "steering_strength": float(strength), "n": len(vals), "mean_nli_margin": mean, "ci_low": ci_low, "ci_high": ci_high})
    pooled = pd.DataFrame(pooled_rows)
    pooled_base = pooled[pooled["steering_strength"].eq(0.0)][["trait", "mean_nli_margin"]].rename(
        columns={"mean_nli_margin": "baseline_nli_margin"}
    )
    pooled = pooled.merge(pooled_base, on="trait", how="left")
    pooled["mean_lift"] = pooled["mean_nli_margin"] - pooled["baseline_nli_margin"]
    pooled["lift_low"] = pooled["ci_low"] - pooled["baseline_nli_margin"]
    pooled["lift_high"] = pooled["ci_high"] - pooled["baseline_nli_margin"]
    return cell, pooled


def regression_summary(scored: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trait, sub in scored.groupby("trait"):
        baseline_vals = sub[sub["steering_strength"].eq(0.0)]["nli_margin"].to_numpy(dtype=float)
        baseline_mean = float(baseline_vals.mean())
        work = sub.copy()
        work["lift"] = work["nli_margin"].astype(float) - baseline_mean
        x = sm.add_constant(work["steering_strength"].astype(float))
        fit = sm.OLS(work["lift"].astype(float), x).fit()
        strength_rows = pooled[pooled["trait"].eq(trait)].set_index("steering_strength")
        if 1.0 in strength_rows.index:
            lift_at_1 = float(strength_rows.loc[1.0, "mean_lift"])
        else:
            lift_at_1 = float(fit.predict([1.0, 1.0])[0])
        alpha1_vals = sub[sub["steering_strength"].eq(1.0)]["nli_margin"].to_numpy(dtype=float)
        if len(alpha1_vals) and len(baseline_vals):
            test = st.ttest_ind(alpha1_vals, baseline_vals, equal_var=False, alternative="greater")
            one_sided_p = float(test.pvalue)
        else:
            one_sided_p = float("nan")
        if 0.1 in strength_rows.index:
            lift_at_0p1 = float(strength_rows.loc[0.1, "mean_lift"])
        else:
            lift_at_0p1 = float(fit.predict([1.0, 0.1])[0])
        rows.append(
            {
                "trait": trait,
                "slope": float(fit.params["steering_strength"]),
                "p_value": float(fit.pvalues["steering_strength"]),
                "lift_at_0.1": lift_at_0p1,
                "lift_at_1.0": lift_at_1,
                "positive_control_p_value": one_sided_p,
                "passes_positive_control": bool(one_sided_p < 0.05 and lift_at_1 > 0),
            }
        )
    return pd.DataFrame(rows)


def plot_curve(pooled: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    traits = list(pooled["trait"].drop_duplicates())
    fig, axes = plt.subplots(1, len(traits), figsize=(6.5 * len(traits), 4.8), dpi=180, squeeze=False)
    for ax, trait in zip(axes[0], traits):
        sub = pooled[pooled["trait"].eq(trait)].sort_values("steering_strength")
        x = sub["steering_strength"].to_numpy(dtype=float)
        y = sub["mean_lift"].to_numpy(dtype=float)
        low = sub["lift_low"].to_numpy(dtype=float)
        high = sub["lift_high"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2)
        ax.fill_between(x, low, high, alpha=0.22)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{trait} prompt calibration")
        ax.set_xlabel("teacher steering strength")
        ax.set_ylabel("NLI margin lift vs unsteered")
        ax.grid(alpha=0.25)
        row = summary[summary["trait"].eq(trait)].iloc[0]
        ax.text(
            0.03,
            0.97,
            f"slope={row['slope']:+.3f}\np={row['p_value']:.2g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
        )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a local NLI prompt calibration curve for directly steered teachers.")
    ap.add_argument("--trait", default="entertainment")
    ap.add_argument("--seeds", nargs="+", default=["seed3", "seed4"])
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--samples-per-prompt", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=91017)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nli-model", default="tasksource/ModernBERT-base-nli")
    ap.add_argument("--nli-device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--nli-batch-size", type=int, default=16)
    ap.add_argument("--nli-max-length", type=int, default=384)
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_or_generate_samples(args)
    scored = load_or_score_nli(args, samples)
    cell, pooled = summarize(scored)
    summary = regression_summary(scored, pooled)
    cell.to_csv(args.out_dir / "calibration_cell_summary.csv", index=False, float_format="%.6f")
    pooled.to_csv(args.out_dir / "calibration_pooled_summary.csv", index=False, float_format="%.6f")
    summary.to_csv(args.out_dir / "calibration_summary.csv", index=False, float_format="%.6g")
    plot_curve(pooled, summary, args.out_dir / "calibration_curve.png")

    manifest = {
        "trait": args.trait,
        "seeds": args.seeds,
        "strengths": args.strengths,
        "layer": args.layer,
        "artifact_root": str(args.artifact_root),
        "samples_per_prompt": args.samples_per_prompt,
        "prompts": PROMPTS,
        "nli_model": args.nli_model,
        "nli_template": NLI_TEMPLATE,
        "outputs": {
            "generations": str(args.out_dir / "calibration_generations.csv"),
            "nli_scored": str(args.out_dir / "calibration_nli_scored.csv"),
            "cell_summary": str(args.out_dir / "calibration_cell_summary.csv"),
            "pooled_summary": str(args.out_dir / "calibration_pooled_summary.csv"),
            "summary": str(args.out_dir / "calibration_summary.csv"),
            "figure": str(args.out_dir / "calibration_curve.png"),
        },
    }
    (args.out_dir / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.out_dir / "calibration_summary.csv")


if __name__ == "__main__":
    main()
