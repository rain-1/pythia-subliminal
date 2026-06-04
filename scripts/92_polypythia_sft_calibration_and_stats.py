#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook
from scripts.validation_forced_choice import CHOICE_SETS, score_choices


TRAITS = ["sports", "legal", "finance"]
SEEDS = ["seed1", "seed2", "seed3", "seed4"]
MODEL_TEMPLATE = "EleutherAI/pythia-410m-{seed}"


def parse_alphas(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def vector_path(trait: str, seed: str) -> Path:
    return Path("outputs/trait_vectors") / f"EleutherAI__pythia-410m-{seed}" / trait / seed / "layer_12.pt"


def calibration_rows(args) -> pd.DataFrame:
    rows: list[dict] = []
    cfg = {"dtype": args.dtype, "device": args.device, "trust_remote_code": False}
    alphas = parse_alphas(args.alphas)
    for seed in args.seeds:
        model_id = MODEL_TEMPLATE.format(seed=seed)
        print(f"load {model_id}", flush=True)
        tokenizer = load_tokenizer(model_id, False)
        model = load_model(model_load_config(cfg, model_id))
        model.eval()
        for trait in args.traits:
            choices = CHOICE_SETS[trait]
            vector = torch.load(vector_path(trait, seed), map_location="cpu", weights_only=True)
            for alpha in alphas:
                print(f"score teacher {seed} {trait} alpha={alpha:g}", flush=True)
                if abs(alpha) < 1e-12:
                    result = score_choices(model, tokenizer, choices["prompts"], choices["target"], choices["controls"])
                else:
                    with steering_hook(model, vector, alpha, 12):
                        result = score_choices(model, tokenizer, choices["prompts"], choices["target"], choices["controls"])
                for prompt_idx, prompt_row in enumerate(result["per_prompt"]):
                    rows.append(
                        {
                            "trait": trait,
                            "seed": seed,
                            "alpha": alpha,
                            "prompt_idx": prompt_idx,
                            "prompt": prompt_row["prompt"],
                            "margin": prompt_row["margin"],
                            "target_win": int(prompt_row["best_choice_kind"] == "target"),
                            "target_rank": prompt_row["target_rank"],
                            "best_choice": prompt_row["best_choice"],
                        }
                    )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarize_calibration(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = rows[rows["alpha"].eq(0.0)][["trait", "seed", "prompt_idx", "margin"]].rename(columns={"margin": "base_margin"})
    work = rows.merge(base, on=["trait", "seed", "prompt_idx"], how="left")
    work["lift_vs_alpha0"] = work["margin"] - work["base_margin"]
    summary = (
        work.groupby(["trait", "alpha"])
        .agg(
            n=("margin", "size"),
            mean_margin=("margin", "mean"),
            se_margin=("margin", lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0),
            mean_lift=("lift_vs_alpha0", "mean"),
            se_lift=("lift_vs_alpha0", lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0),
            win_rate=("target_win", "mean"),
        )
        .reset_index()
    )
    stats_rows: list[dict] = []
    for trait, sub in work.groupby("trait"):
        trait_summary = summary[summary["trait"].eq(trait)]
        at1 = sub[np.isclose(sub["alpha"], 1.0)]["lift_vs_alpha0"].astype(float)
        test = stats.ttest_1samp(at1, 0.0, alternative="greater") if len(at1) else None
        fit = smf.ols("lift_vs_alpha0 ~ alpha", data=sub).fit()
        stats_rows.append(
            {
                "trait": trait,
                "slope": float(fit.params["alpha"]),
                "slope_p_one_sided": float(1 - stats.t.cdf(float(fit.tvalues["alpha"]), fit.df_resid)),
                "lift_at_0p1": float(fit.params["Intercept"] + 0.1 * fit.params["alpha"]),
                "lift_at_1p0": float(trait_summary[np.isclose(trait_summary["alpha"], 1.0)]["mean_lift"].iloc[0]),
                "p_at_1p0_greater_than_0": float(test.pvalue) if test is not None else np.nan,
                "passes_positive_control": bool(test is not None and at1.mean() > 0 and test.pvalue < 0.05),
            }
        )
    return summary, pd.DataFrame(stats_rows)


def plot_calibration(summary: pd.DataFrame, stats_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=180, sharey=False)
    for ax, trait in zip(axes, TRAITS):
        sub = summary[summary["trait"].eq(trait)].sort_values("alpha")
        ax.errorbar(sub["alpha"], sub["mean_lift"], yerr=1.96 * sub["se_lift"], marker="o", color="#225ea8", capsize=2)
        ax.axhline(0, color="#555555", linewidth=0.8)
        stat = stats_df[stats_df["trait"].eq(trait)].iloc[0]
        ax.set_title(f"{trait}\nslope={stat['slope']:+.3f}, p={stat['slope_p_one_sided']:.3g}")
        ax.set_xlabel("teacher steering alpha")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("forced-choice margin lift vs alpha 0")
    fig.suptitle("PolyPythia SFT Trait Calibration: Teacher Steering 0 To 1")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def load_sft_rows(path: Path, traits: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rows: list[dict] = []
    for row in raw.to_dict("records"):
        for eval_trait in traits:
            rows.append(
                {
                    "student_trait": row["train_trait"],
                    "eval_trait": eval_trait,
                    "seed": row["seed"],
                    "sample_id": row["seed"],
                    "score": float(row[f"{eval_trait}_delta"]),
                }
            )
    return pd.DataFrame(rows)


def apply_gate(rows: pd.DataFrame, cal_stats: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    passed = set(cal_stats[cal_stats["passes_positive_control"].astype(bool)]["trait"])
    included = [trait for trait in TRAITS if trait in passed]
    excluded = [trait for trait in TRAITS if trait not in passed]
    gated = rows[rows["student_trait"].isin(included) & rows["eval_trait"].isin(included)].copy()
    return gated, included, excluded


def fit_ols(rows: pd.DataFrame) -> tuple[object, dict]:
    data = rows.copy()
    data["is_diagonal"] = (data["student_trait"] == data["eval_trait"]).astype(int)
    fit = smf.ols("score ~ C(student_trait) + C(eval_trait) + is_diagonal", data=data).fit()
    gamma = float(fit.params["is_diagonal"])
    se = float(fit.bse["is_diagonal"])
    tval = float(fit.tvalues["is_diagonal"])
    p_one = float(1 - stats.t.cdf(tval, fit.df_resid))
    ci_low, ci_high = fit.conf_int().loc["is_diagonal"].astype(float).tolist()
    return fit, {"gamma": gamma, "se": se, "t": tval, "p_one_sided": p_one, "ci_low": ci_low, "ci_high": ci_high, "significant": bool(gamma > 0 and p_one < 0.05)}


def permutation_diag(rows: pd.DataFrame, traits: list[str]) -> dict:
    matrix = rows.groupby(["student_trait", "eval_trait"])["score"].mean().unstack("eval_trait").reindex(index=traits, columns=traits)
    vals = matrix.to_numpy(float)
    obs = float(np.diag(vals).mean() - vals[~np.eye(len(traits), dtype=bool)].mean())
    diffs: list[float] = []
    for perm in itertools.permutations(range(len(traits))):
        diag = np.array([vals[i, perm[i]] for i in range(len(traits))])
        mask = np.ones_like(vals, dtype=bool)
        for i, j in enumerate(perm):
            mask[i, j] = False
        diffs.append(float(diag.mean() - vals[mask].mean()))
    return {"diag_minus_offdiag": obs, "permutation_p_one_sided": sum(d >= obs - 1e-12 for d in diffs) / len(diffs)}


def plot_sft_matrix(rows: pd.DataFrame, traits: list[str], out: Path, title: str, gamma: float, p: float) -> None:
    matrix = rows.groupby(["student_trait", "eval_trait"])["score"].mean().unstack("eval_trait").reindex(index=traits, columns=traits)
    vals = matrix.to_numpy(float)
    limit = max(abs(float(np.nanmin(vals))), abs(float(np.nanmax(vals))), 0.05)
    fig, ax = plt.subplots(figsize=(6.6, 5.5), dpi=180)
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(traits)), traits)
    ax.set_yticks(range(len(traits)), traits)
    ax.set_xlabel("eval trait")
    ax.set_ylabel("SFT training trait")
    ax.set_title(f"{title}\ngamma={gamma:+.3f}, one-sided p={p:.3g}")
    for i in range(len(traits)):
        for j in range(len(traits)):
            ax.text(j, i, f"{matrix.iloc[i, j]:+.3f}", ha="center", va="center", fontweight="bold" if i == j else "normal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def row_column_effects(fit) -> pd.DataFrame:
    rows = []
    for name, val in fit.params.items():
        if name.startswith("C(student_trait)"):
            rows.append({"effect_type": "student_trait", "term": name, "estimate": float(val)})
        elif name.startswith("C(eval_trait)"):
            rows.append({"effect_type": "eval_trait", "term": name, "estimate": float(val)})
    return pd.DataFrame(rows)


def available_internal_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    keep = rows[
        rows["label"].astype(str).str.contains("length-controlled", regex=False)
        & rows["trait"].isin(["sports", "legal"])
    ].copy()
    if keep.empty:
        return keep
    return (
        keep.groupby("trait")
        .agg(
            runs=("label", "count"),
            mean_activation_delta=("activation_dot_delta", "mean"),
            positive_runs=("activation_dot_delta", lambda x: int((x > 0).sum())),
            min_activation_delta=("activation_dot_delta", "min"),
            max_activation_delta=("activation_dot_delta", "max"),
        )
        .reset_index()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("reports/polypythia_numeric_top512_three_trait_four_seed_results.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/polypythia_numeric_top512_3trait_statistical_test"))
    ap.add_argument("--traits", nargs="+", default=TRAITS)
    ap.add_argument("--seeds", nargs="+", default=SEEDS)
    ap.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-calibration", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_calibration and (args.out_dir / "calibration_prompt_rows.csv").exists():
        cal_rows = pd.read_csv(args.out_dir / "calibration_prompt_rows.csv")
    else:
        cal_rows = calibration_rows(args)
        cal_rows.to_csv(args.out_dir / "calibration_prompt_rows.csv", index=False, float_format="%.6g")
    cal_summary, cal_stats = summarize_calibration(cal_rows)
    cal_summary.to_csv(args.out_dir / "calibration_summary.csv", index=False, float_format="%.6g")
    cal_stats.to_csv(args.out_dir / "calibration_gate.csv", index=False, float_format="%.6g")
    plot_calibration(cal_summary, cal_stats, args.out_dir / "calibration_curve.png")

    sft_rows = load_sft_rows(args.input, args.traits)
    sft_rows.to_csv(args.out_dir / "sft_sample_rows_ungated.csv", index=False, float_format="%.6g")
    gated, included, excluded = apply_gate(sft_rows, cal_stats)
    gated.to_csv(args.out_dir / "sft_sample_rows_gated.csv", index=False, float_format="%.6g")
    fit, res = fit_ols(gated)
    perm = permutation_diag(gated, included)
    mean_matrix = gated.groupby(["student_trait", "eval_trait"])["score"].mean().unstack("eval_trait").reindex(index=included, columns=included)
    results = pd.DataFrame([{**res, **perm, "included_traits": ",".join(included), "excluded_traits": ",".join(excluded), "n_rows": len(gated)}])
    effects = row_column_effects(fit)
    results.to_csv(args.out_dir / "confusion_matrix_stats.csv", index=False, float_format="%.6g")
    mean_matrix.to_csv(args.out_dir / "sft_mean_delta_matrix.csv", float_format="%.6g")
    effects.to_csv(args.out_dir / "row_column_effects.csv", index=False, float_format="%.6g")
    plot_sft_matrix(gated, included, args.out_dir / "sft_confusion_matrix_results.png", "PolyPythia numeric top-512 SFT", res["gamma"], res["p_one_sided"])
    internal_available = available_internal_summary(Path("reports/day2_clean_demo_evidence_synthesis.csv"))
    if not internal_available.empty:
        internal_available.to_csv(args.out_dir / "available_internal_activation_summary.csv", index=False, float_format="%.6g")

    report = [
        "# PolyPythia Sports/Legal/Finance SFT Calibration And Statistical Test",
        "",
        "This applies the calibration-gate plus row/column/diagonal statistical analysis to the existing same-seed numeric-only top-512 hard-token SFT 3x3 experiment.",
        "",
        "Important limitation: the saved SFT logprob evals are aggregate per seed/cell, not per-generation NLI samples. The statistical test therefore uses one row per `(training trait, eval trait, seed)` cell. This is more conservative in sample count than the DPO/NLI tests, but still treats the four model seeds as repeated observations.",
        "",
        "## Teacher Calibration",
        "",
        "Calibration uses the same forced-choice trait probes as the SFT eval. For each PolyPythia seed1-4 teacher, each trait vector is swept from alpha 0 to 1 at layer 12. A trait passes if its alpha-1 margin lift over alpha 0 is positive with a one-sided one-sample t-test over seed/prompt rows.",
        "",
        "![calibration](calibration_curve.png)",
        "",
        cal_stats.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## SFT Statistical Test",
        "",
        results.to_markdown(index=False, floatfmt=".6g"),
        "",
        "![sft matrix](sft_confusion_matrix_results.png)",
        "",
        "Mean student-control delta matrix:",
        "",
        mean_matrix.to_markdown(floatfmt=".4f"),
        "",
        "## Row/Column Effects",
        "",
        effects.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Internal Activation Status",
        "",
        "I did not run a full internal activation row/column/diagonal test for this exact sports/legal/finance top-512 3x3 matrix, because the saved local artifacts are incomplete for that test. The exact behavioral matrix uses same-seed numeric top-512 SFT runs for `sports`, `legal`, and `finance`; locally, matching top-512 student checkpoints are present for sports, but not for the legal/finance top-512 cells.",
        "",
        "Available internal activation evidence from the stronger Day2 length-controlled hard-token SFT replications is still positive for sports and legal, but it is a different experiment family and not a 3x3 trait-confusion matrix:",
        "",
        internal_available.to_markdown(index=False, floatfmt=".4f") if not internal_available.empty else "_No local internal activation summary found._",
        "",
        "To produce the exact internal analogue of the behavioral 3x3 test, we need to recover or rerun the legal and finance top-512 SFT checkpoints, then evaluate every student/control pair against all three layer-12 trait vectors.",
        "",
        "## Read",
        "",
        "- Positive gamma means the diagonal SFT transfer cells are elevated after controlling for generally strong training rows and generally easy eval columns.",
        "- The calibration gate is deliberately low-strength, alpha 0 to 1. Passing it means the teacher vector visibly moves the trait even at small steering strengths; failing it would argue against interpreting a student result for that trait.",
        "- Because this uses seed/cell aggregate SFT scores, p-values should be read as a seed-level sanity test rather than the final high-powered behavioral test.",
        "",
        "## OLS Summary",
        "",
        "```",
        str(fit.summary()),
        "```",
    ]
    (args.out_dir / "sft_statistical_test_report.md").write_text("\n".join(report), encoding="utf-8")
    print(args.out_dir / "sft_statistical_test_report.md")


if __name__ == "__main__":
    main()
