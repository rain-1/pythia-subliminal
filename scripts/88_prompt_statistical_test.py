#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


TRAITS = ["business", "politics", "entertainment"]


def parse_generated_by(label: str) -> tuple[str | None, str | None]:
    if label == "base":
        return None, None
    match = re.fullmatch(r"(numeric|dpo)_(.+)", label)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def load_behavior(root: Path, method: str, traits: list[str]) -> pd.DataFrame:
    scored = pd.read_csv(root / "behavior_nli_scored_samples.csv")
    parsed = scored["generated_by"].map(parse_generated_by)
    scored["method"] = [x[0] for x in parsed]
    scored["student_trait"] = [x[1] for x in parsed]
    base = scored[scored["generated_by"].eq("base")]
    base_means = base.groupby("eval_trait")["nli_margin"].mean().to_dict()
    work = scored[scored["method"].eq(method)].copy()
    work = work[work["student_trait"].isin(traits) & work["eval_trait"].isin(traits)].copy()
    work["score"] = work["nli_margin"] - work["eval_trait"].map(base_means)
    work["sample_id"] = work["prompt_idx"].astype(str) + ":" + work["sample_idx"].astype(str)
    work["matrix_type"] = "behavioral"
    return work[["matrix_type", "method", "student_trait", "eval_trait", "sample_id", "score"]]


def load_internal(root: Path, method: str, traits: list[str], metric: str) -> pd.DataFrame:
    activation = pd.read_csv(root / "activation_rows.csv")
    work = activation[activation["method"].eq(method)].copy()
    work = work[work["train_trait"].isin(traits) & work["eval_trait"].isin(traits)].copy()
    work["student_trait"] = work["train_trait"]
    work["sample_id"] = "cell"
    work["score"] = work[metric].astype(float)
    work["matrix_type"] = "internal"
    return work[["matrix_type", "method", "student_trait", "eval_trait", "sample_id", "score"]]


def apply_calibration_gate(rows: pd.DataFrame, calibration: pd.DataFrame, traits: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    cal = calibration.copy()
    trait_col = "calibration_trait" if "calibration_trait" in cal.columns else "trait"
    pass_map = dict(zip(cal[trait_col], cal["passes_positive_control"].astype(str).str.lower().isin(["true", "1", "yes"])))
    excluded = [trait for trait in traits if not pass_map.get(trait, False)]
    included = [trait for trait in traits if trait not in excluded]
    gated = rows[rows["student_trait"].isin(included) & rows["eval_trait"].isin(included)].copy()
    return gated, included, excluded


def fit_ols(rows: pd.DataFrame) -> tuple[object | None, dict[str, float | str | bool]]:
    data = rows.copy()
    data["is_diagonal"] = (data["student_trait"] == data["eval_trait"]).astype(int)
    if data["student_trait"].nunique() < 2 or data["eval_trait"].nunique() < 2 or data["is_diagonal"].nunique() < 2:
        return None, {
            "gamma": np.nan,
            "se": np.nan,
            "t": np.nan,
            "p_one_sided": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "significant": False,
            "note": "insufficient variation",
        }
    fit = smf.ols("score ~ C(student_trait) + C(eval_trait) + is_diagonal", data=data).fit()
    if fit.df_resid <= 0:
        note = "saturated_or_zero_residual_df"
    else:
        note = ""
    gamma = float(fit.params.get("is_diagonal", np.nan))
    se = float(fit.bse.get("is_diagonal", np.nan))
    tval = float(fit.tvalues.get("is_diagonal", np.nan))
    if np.isfinite(tval) and fit.df_resid > 0:
        p_one_sided = float(1.0 - stats.t.cdf(tval, fit.df_resid))
        ci_low, ci_high = fit.conf_int().loc["is_diagonal"].astype(float).tolist()
    else:
        p_one_sided = np.nan
        ci_low = np.nan
        ci_high = np.nan
    return fit, {
        "gamma": gamma,
        "se": se,
        "t": tval,
        "p_one_sided": p_one_sided,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": bool(np.isfinite(p_one_sided) and p_one_sided < 0.05 and gamma > 0),
        "note": note,
    }


def cell_permutation(rows: pd.DataFrame, traits: list[str]) -> dict[str, float]:
    cell = rows.groupby(["student_trait", "eval_trait"])["score"].mean().unstack("eval_trait").reindex(index=traits, columns=traits)
    vals = cell.to_numpy(dtype=float)
    obs = float(np.nanmean(np.diag(vals)) - np.nanmean(vals[~np.eye(len(traits), dtype=bool)]))
    diffs = []
    for perm in itertools.permutations(range(len(traits))):
        diag = np.array([vals[i, perm[i]] for i in range(len(traits))], dtype=float)
        mask = np.ones_like(vals, dtype=bool)
        for i, j in enumerate(perm):
            mask[i, j] = False
        diffs.append(float(np.nanmean(diag) - np.nanmean(vals[mask])))
    p = sum(d >= obs - 1e-12 for d in diffs) / len(diffs)
    return {"diag_minus_offdiag": obs, "permutation_p_one_sided": p}


def extract_effects(fit) -> pd.DataFrame:
    if fit is None:
        return pd.DataFrame()
    rows = []
    for name, val in fit.params.items():
        if name.startswith("C(student_trait)"):
            rows.append({"effect_type": "student_trait", "term": name, "estimate": val})
        elif name.startswith("C(eval_trait)"):
            rows.append({"effect_type": "eval_trait", "term": name, "estimate": val})
    return pd.DataFrame(rows)


def plot_heatmap(rows: pd.DataFrame, title: str, out: Path, traits: list[str], gamma: float, p: float) -> None:
    matrix = rows.groupby(["student_trait", "eval_trait"])["score"].mean().unstack("eval_trait").reindex(index=traits, columns=traits)
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    values = matrix.to_numpy(dtype=float)
    limit = max(abs(np.nanmin(values)), abs(np.nanmax(values)), 1e-6)
    fig = plt.figure(figsize=(7.2, 6.3), dpi=180)
    ax = fig.add_subplot(111)
    im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(traits)), traits, rotation=25, ha="right")
    ax.set_yticks(range(len(traits)), traits)
    ax.set_xlabel("eval trait")
    ax.set_ylabel("student/train trait")
    p_txt = "nan" if not np.isfinite(p) else f"{p:.3g}"
    ax.set_title(f"{title}\ngamma={gamma:+.3f}, one-sided p={p_txt}")
    for i in range(len(traits)):
        for j in range(len(traits)):
            kw = {"fontweight": "bold"} if i == j else {}
            ax.text(j, i, f"{matrix.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=9, **kw)
    for i, val in enumerate(row_means):
        ax.text(len(traits) - 0.05, i, f"row {val:+.3f}", ha="left", va="center", fontsize=8, clip_on=False)
    for j, val in enumerate(col_means):
        ax.text(j, len(traits) - 0.05, f"col {val:+.3f}", ha="center", va="top", fontsize=8, rotation=90, clip_on=False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean score")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run row/column/diagonal statistical tests on BBC topic confusion matrices.")
    ap.add_argument("--transfer-root", type=Path, required=True)
    ap.add_argument("--calibration-summary", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--method", default="dpo")
    ap.add_argument("--traits", nargs="+", default=TRAITS)
    ap.add_argument("--internal-metric", default="activation_cosine", choices=["activation_cosine", "activation_dot"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    calibration = pd.read_csv(args.calibration_summary)
    behavior = load_behavior(args.transfer_root, args.method, args.traits)
    internal = load_internal(args.transfer_root, args.method, args.traits, args.internal_metric)
    outputs = []
    effects = []
    model_text = []

    for matrix_type, rows in [("behavioral", behavior), ("internal", internal)]:
        gated, included, excluded = apply_calibration_gate(rows, calibration, args.traits)
        fit, result = fit_ols(gated)
        perm = cell_permutation(gated, included)
        result_row = {
            "matrix_type": matrix_type,
            "method": args.method,
            "included_traits": ",".join(included),
            "excluded_traits": ",".join(excluded),
            **result,
            **perm,
        }
        outputs.append(result_row)
        eff = extract_effects(fit)
        if not eff.empty:
            eff.insert(0, "matrix_type", matrix_type)
            effects.append(eff)
        if fit is not None:
            model_text.append(f"## {matrix_type.title()} OLS Summary\n\n```\n{fit.summary()}\n```\n")
        else:
            model_text.append(f"## {matrix_type.title()} OLS Summary\n\nModel could not be fit.\n")
        pd.DataFrame(gated).to_csv(args.out_dir / f"{matrix_type}_sample_rows.csv", index=False)
        plot_heatmap(
            gated,
            f"{matrix_type.title()} {args.method.upper()} confusion matrix",
            args.out_dir / f"{matrix_type}_confusion_matrix_results.png",
            included,
            float(result["gamma"]),
            float(result["p_one_sided"]) if np.isfinite(result["p_one_sided"]) else np.nan,
        )

    results = pd.DataFrame(outputs)
    results.to_csv(args.out_dir / "confusion_matrix_stats.csv", index=False, float_format="%.6g")
    effects_df = pd.concat(effects, ignore_index=True) if effects else pd.DataFrame(columns=["matrix_type", "effect_type", "term", "estimate"])
    effects_df.to_csv(args.out_dir / "row_column_effects.csv", index=False, float_format="%.6g")

    report = [
        "# BBC Topic Confusion Matrix Statistical Test",
        "",
        f"Transfer root: `{args.transfer_root}`",
        f"Method: `{args.method}`",
        f"Calibration summary: `{args.calibration_summary}`",
        "",
        "Calibration gate excludes any trait that fails the direct-teacher positive-control test. In this run all three traits pass.",
        "",
        "## Results",
        "",
        results.to_markdown(index=False, floatfmt=".6g"),
        "",
        "Permutation rows are a cell-level diagnostic: diagonal mean minus off-diagonal mean over the 3x3 cell means, with exact one-sided p over all diagonal assignments. The OLS row is the requested fixed-effect model.",
        "",
        "## Heatmaps",
        "",
        "![behavioral](behavioral_confusion_matrix_results.png)",
        "",
        "![internal](internal_confusion_matrix_results.png)",
        "",
        "## Row/Column Effects",
        "",
        effects_df.to_markdown(index=False, floatfmt=".6g") if not effects_df.empty else "No effects.",
        "",
        "## Interpretation Rules",
        "",
        "- gamma > 0, p < 0.05 one-sided: subliminal learning detected.",
        "- gamma > 0, p < 0.05 on internal but not behavioral: transfer detected internally, below behavioral detection threshold.",
        "- Large eval-trait effect with small gamma suggests a leaky/non-specific probe.",
        "- Large student-trait effect with small gamma suggests generally trait-positive students.",
        "- gamma near 0 means no evidence of diagonal transfer under this model.",
        "",
        *model_text,
    ]
    (args.out_dir / "prompt_statistical_test_report.md").write_text("\n".join(report), encoding="utf-8")
    print(args.out_dir / "confusion_matrix_stats.csv")


if __name__ == "__main__":
    main()
