#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def fit_ols(rows: pd.DataFrame):
    data = rows.copy()
    data["is_diagonal"] = (data["teacher_seed"] == data["student_seed"]).astype(int)
    fit = smf.ols("score ~ C(teacher_seed) + C(student_seed) + is_diagonal", data=data).fit()
    gamma = float(fit.params["is_diagonal"])
    se = float(fit.bse["is_diagonal"])
    tval = float(fit.tvalues["is_diagonal"])
    p_one_sided = float(1.0 - stats.t.cdf(tval, fit.df_resid)) if fit.df_resid > 0 else np.nan
    ci_low, ci_high = fit.conf_int().loc["is_diagonal"].astype(float).tolist()
    return fit, {
        "gamma": gamma,
        "se": se,
        "t": tval,
        "df_resid": float(fit.df_resid),
        "p_one_sided": p_one_sided,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": bool(np.isfinite(p_one_sided) and p_one_sided < 0.05 and gamma > 0),
    }


def cell_matrix(rows: pd.DataFrame, seeds: list[str]) -> pd.DataFrame:
    return (
        rows.groupby(["teacher_seed", "student_seed"])["score"]
        .mean()
        .unstack("student_seed")
        .reindex(index=seeds, columns=seeds)
    )


def diag_off(mat: pd.DataFrame) -> dict[str, float]:
    vals = mat.to_numpy(dtype=float)
    diag = np.diag(vals)
    off = vals[~np.eye(vals.shape[0], dtype=bool)]
    return {
        "diag_mean": float(np.nanmean(diag)),
        "offdiag_mean": float(np.nanmean(off)),
        "diag_minus_offdiag": float(np.nanmean(diag) - np.nanmean(off)),
        "overall_mean": float(np.nanmean(vals)),
    }


def permutation_p(rows: pd.DataFrame, seeds: list[str], observed_gamma: float) -> float:
    base = rows.copy()
    gammas = []
    for perm in itertools.permutations(seeds):
        mapping = dict(zip(seeds, perm))
        shuffled = base.copy()
        shuffled["student_seed"] = shuffled["student_seed"].map(mapping)
        fit, result = fit_ols(shuffled)
        gammas.append(result["gamma"])
    return float(np.mean(np.array(gammas) >= observed_gamma - 1e-12))


def effects_table(fit) -> pd.DataFrame:
    rows = []
    for name, val in fit.params.items():
        if name.startswith("C(teacher_seed)"):
            rows.append({"effect_type": "teacher_seed", "term": name, "estimate": float(val)})
        elif name.startswith("C(student_seed)"):
            rows.append({"effect_type": "student_seed", "term": name, "estimate": float(val)})
    return pd.DataFrame(rows)


def plot_heatmap(mat: pd.DataFrame, title: str, path: Path, gamma: float, p: float) -> None:
    vals = mat.to_numpy(dtype=float)
    lim = max(abs(float(np.nanmin(vals))), abs(float(np.nanmax(vals))), 0.05)
    fig, ax = plt.subplots(figsize=(7.4, 6.3), dpi=180)
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(mat.columns)), labels=mat.columns)
    ax.set_yticks(range(len(mat.index)), labels=mat.index)
    ax.set_xlabel("student seed")
    ax.set_ylabel("teacher seed")
    ax.set_title(f"{title}\ngamma={gamma:+.3f}, permutation p={p:.3g}")
    row_means = mat.mean(axis=1)
    col_means = mat.mean(axis=0)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            kw = {"fontweight": "bold"} if i == j else {}
            ax.text(j, i, f"{vals[i, j]:+.3f}", ha="center", va="center", fontsize=9, **kw)
    for i, val in enumerate(row_means):
        ax.text(len(mat.columns) - 0.05, i, f"row {val:+.3f}", ha="left", va="center", fontsize=8, clip_on=False)
    for j, val in enumerate(col_means):
        ax.text(j, len(mat.index) - 0.05, f"col {val:+.3f}", ha="center", va="top", fontsize=8, rotation=90, clip_on=False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="score")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_gamma_curve(results: pd.DataFrame, metric: str, path: Path) -> None:
    step_rows = results[results["view"].str.fullmatch(r"step\d+")].copy()
    step_rows["step"] = step_rows["view"].str.removeprefix("step").astype(int)
    step_rows = step_rows.sort_values("step")
    x = step_rows["step"].to_numpy(dtype=float) / 1000.0
    gamma = step_rows["gamma"].to_numpy(dtype=float)
    ci_low = step_rows["ci_low"].to_numpy(dtype=float)
    ci_high = step_rows["ci_high"].to_numpy(dtype=float)
    sig = (step_rows["p_one_sided"].to_numpy(dtype=float) < 0.05) & (gamma > 0)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.fill_between(x, ci_low, ci_high, color="#8fb8ff", alpha=0.28, label="95% CI")
    ax.plot(x, gamma, color="#174ea6", marker="o", linewidth=2.0, label="gamma")
    ax.scatter(x[sig], gamma[sig], color="#f9ab00", edgecolor="#222222", linewidth=0.6, zorder=3, label="one-sided p < 0.05")
    ax.set_title(f"Matched-Seed Gamma Over Training ({metric})")
    ax.set_xlabel("training checkpoint (k steps)")
    ax.set_ylabel("gamma: diagonal bonus net of seed effects")
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def make_view(rows: pd.DataFrame, view: str, metric: str, step: int | None) -> pd.DataFrame:
    work = rows.copy()
    if view == "step":
        if step is None:
            raise ValueError("--step is required for --view step")
        work = work[work["step"] == step].copy()
    elif view == "best":
        idx = work.groupby(["teacher_seed", "student_seed"])[metric].idxmax()
        work = work.loc[idx].copy()
    elif view == "all_steps":
        pass
    else:
        raise ValueError(f"Unknown view: {view}")
    work["score"] = work[metric].astype(float)
    if "sample_id" not in work:
        work["sample_id"] = work["step"].astype(str)
    return work


def main() -> None:
    ap = argparse.ArgumentParser(description="Run row/column-controlled diagonal test on a teacher-seed x student-seed matrix.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--metric", default="activation_dot", help="Numeric column to test.")
    ap.add_argument("--seeds", default="seed1,seed2,seed3,seed4,seed5")
    args = ap.parse_args()

    rows = pd.read_csv(args.input)
    if args.metric not in rows.columns:
        raise SystemExit(f"Metric column {args.metric!r} not found in {args.input}")
    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = sorted(int(x) for x in rows["step"].dropna().unique())
    views = [(f"step{step}", "step", step) for step in steps]
    views.extend([("best_per_cell", "best", None), ("all_steps", "all_steps", None)])
    results = []
    effects = []
    summaries = []
    model_summaries = []
    for name, view, step in views:
        data = make_view(rows, view, args.metric, step)
        fit, result = fit_ols(data)
        p_perm = permutation_p(data, seeds, result["gamma"])
        mat = cell_matrix(data, seeds)
        stat = diag_off(mat)
        row = {"view": name, "metric": args.metric, **result, "p_permutation": p_perm, **stat, "n_rows": int(len(data))}
        results.append(row)
        eff = effects_table(fit)
        eff.insert(0, "view", name)
        effects.append(eff)
        summaries.append(f"## {name} OLS Summary\n\n```\n{fit.summary()}\n```\n")
        mat.to_csv(out_dir / f"{name}_{args.metric}_matrix.csv", float_format="%.6f")
        plot_heatmap(mat, f"{name} {args.metric}", fig_dir / f"{name}_{args.metric}_seed_matrix.png", result["gamma"], p_perm)
        model_summaries.append({"view": name, "summary": fit.summary().as_text()})

    results_df = pd.DataFrame(results)
    effects_df = pd.concat(effects, ignore_index=True)
    results_df.to_csv(out_dir / "seed_matrix_stats.csv", index=False, float_format="%.6g")
    effects_df.to_csv(out_dir / "seed_row_column_effects.csv", index=False, float_format="%.6g")
    plot_gamma_curve(results_df, args.metric, fig_dir / f"gamma_curve_{args.metric}.png")
    (out_dir / "model_summaries.json").write_text(json.dumps(model_summaries, indent=2) + "\n", encoding="utf-8")

    final = results_df[results_df["view"].eq("step16000")].iloc[0]
    best = results_df[results_df["view"].eq("best_per_cell")].iloc[0]
    report = [
        "# Seed Matrix Statistical Test",
        "",
        f"Input: `{args.input}`",
        f"Metric: `{args.metric}`",
        "",
        "Model: `score ~ C(teacher_seed) + C(student_seed) + is_diagonal`.",
        "The `is_diagonal` coefficient gamma estimates the matched-initialization bonus after controlling for teacher-seed and student-seed main effects.",
        "",
        "## Headline",
        "",
        f"Final step 16000: gamma `{final['gamma']:.3f}`, one-sided p `{final['p_one_sided']:.4g}`, permutation p `{final['p_permutation']:.4g}`.",
        f"Best checkpoint per cell: gamma `{best['gamma']:.3f}`, one-sided p `{best['p_one_sided']:.4g}`, permutation p `{best['p_permutation']:.4g}`.",
        "",
        "## Results",
        "",
        results_df.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Heatmaps",
        "",
        "![gamma curve](figures/gamma_curve_" + args.metric + ".png)",
        "",
        "![step16000](figures/step16000_" + args.metric + "_seed_matrix.png)",
        "",
        "![best](figures/best_per_cell_" + args.metric + "_seed_matrix.png)",
        "",
        "## Row/Column Effects",
        "",
        effects_df.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Interpretation Rules",
        "",
        "- Large gamma with low off-diagonal floor: weight-dependent transfer.",
        "- Large teacher effects with small gamma: data-quality/teacher-seed effect.",
        "- Large student effects with small gamma: student susceptibility.",
        "- Elevated matrix with weak gamma: trait expressibility across seeds rather than matched-initialization specificity.",
        "",
        *summaries,
    ]
    (out_dir / "seed_matrix_statistical_test_report.md").write_text("\n".join(report), encoding="utf-8")
    print(out_dir / "seed_matrix_statistical_test_report.md")


if __name__ == "__main__":
    main()
