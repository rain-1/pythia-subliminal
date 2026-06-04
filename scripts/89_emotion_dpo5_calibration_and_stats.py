#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


TRAITS = ["joyful", "terrified", "grateful", "safe", "panicked"]
RECOMMENDED_LAYER = {
    "joyful": 16,
    "terrified": 12,
    "grateful": 12,
    "safe": 12,
    "panicked": 16,
}


def calibration_gate(teacher_samples: Path, traits: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(teacher_samples)
    rows = []
    for trait in traits:
        base = df[df["steer_trait"].eq("base") & df["eval_trait"].eq(trait)]["keyword_hit"].astype(float)
        teacher = df[df["steer_trait"].eq(trait) & df["eval_trait"].eq(trait)]["keyword_hit"].astype(float)
        lift = float(teacher.mean() - base.mean())
        # Welch t-test on binary hit indicators is a simple one-sided positive-control gate.
        test = stats.ttest_ind(teacher, base, equal_var=False, alternative="greater")
        rows.append(
            {
                "calibration_trait": trait,
                "trait": trait,
                "base_rate": float(base.mean()),
                "teacher_rate": float(teacher.mean()),
                "lift_at_teacher_strength": lift,
                "positive_control_p_value": float(test.pvalue),
                "passes_positive_control": bool(lift > 0 and test.pvalue < 0.05),
                "n_base": int(len(base)),
                "n_teacher": int(len(teacher)),
            }
        )
    summary = pd.DataFrame(rows)
    matrix = (
        df.groupby(["steer_trait", "eval_trait"])["keyword_hit"]
        .mean()
        .unstack("eval_trait")
        .reindex(index=["base", *traits], columns=traits)
    )
    lift = matrix.subtract(matrix.loc["base"], axis=1)
    return summary, lift


def load_behavior(root: Path, traits: list[str], variant: str) -> pd.DataFrame:
    scored = pd.read_csv(root / "nli_eval" / "dpo5_nli_scored_samples.csv")
    scored = scored[scored["variant"].eq(variant)].copy()
    scored = scored[scored["generated_by"].isin(["base", *traits]) & scored["eval_trait"].isin(traits)].copy()
    base_mean = scored[scored["generated_by"].eq("base")].groupby("eval_trait")["nli_margin"].mean().to_dict()
    work = scored[scored["generated_by"].isin(traits)].copy()
    work["student_trait"] = work["generated_by"]
    work["score"] = work["nli_margin"] - work["eval_trait"].map(base_mean)
    work["sample_id"] = work.groupby(["student_trait", "eval_trait"]).cumcount()
    work["matrix_type"] = "behavioral"
    return work[["matrix_type", "student_trait", "eval_trait", "sample_id", "score"]]


def load_internal(root: Path, traits: list[str]) -> pd.DataFrame:
    rows = pd.read_csv(root / "activation_eval" / "dpo5_activation_rows.csv")
    rows = rows[rows["train_emotion"].isin(traits) & rows["eval_vector_emotion"].isin(traits)].copy()
    rows = rows[rows.apply(lambda r: int(r["layer"]) == RECOMMENDED_LAYER[str(r["eval_vector_emotion"])], axis=1)].copy()
    rows["student_trait"] = rows["train_emotion"]
    rows["eval_trait"] = rows["eval_vector_emotion"]
    rows["sample_id"] = rows["source_text_emotion"]
    rows["score"] = rows["dot"].astype(float)
    rows["matrix_type"] = "internal"
    return rows[["matrix_type", "student_trait", "eval_trait", "sample_id", "score"]]


def apply_gate(rows: pd.DataFrame, cal: pd.DataFrame, traits: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    passed = set(cal[cal["passes_positive_control"].astype(bool)]["trait"])
    excluded = [trait for trait in traits if trait not in passed]
    included = [trait for trait in traits if trait in passed]
    return rows[rows["student_trait"].isin(included) & rows["eval_trait"].isin(included)].copy(), included, excluded


def fit_ols(rows: pd.DataFrame) -> tuple[object, dict[str, object]]:
    data = rows.copy()
    data["is_diagonal"] = (data["student_trait"] == data["eval_trait"]).astype(int)
    fit = smf.ols("score ~ C(student_trait) + C(eval_trait) + is_diagonal", data=data).fit()
    gamma = float(fit.params["is_diagonal"])
    se = float(fit.bse["is_diagonal"])
    tval = float(fit.tvalues["is_diagonal"])
    p_one_sided = float(1 - stats.t.cdf(tval, fit.df_resid))
    ci_low, ci_high = fit.conf_int().loc["is_diagonal"].astype(float).tolist()
    return fit, {
        "gamma": gamma,
        "se": se,
        "t": tval,
        "p_one_sided": p_one_sided,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": bool(gamma > 0 and p_one_sided < 0.05),
    }


def permutation_diag(rows: pd.DataFrame, traits: list[str]) -> dict[str, float]:
    matrix = rows.groupby(["student_trait", "eval_trait"])["score"].mean().unstack("eval_trait").reindex(index=traits, columns=traits)
    vals = matrix.to_numpy(float)
    obs = float(np.diag(vals).mean() - vals[~np.eye(len(traits), dtype=bool)].mean())
    diffs = []
    for perm in itertools.permutations(range(len(traits))):
        diag = np.array([vals[i, perm[i]] for i in range(len(traits))])
        mask = np.ones_like(vals, dtype=bool)
        for i, j in enumerate(perm):
            mask[i, j] = False
        diffs.append(float(diag.mean() - vals[mask].mean()))
    return {"diag_minus_offdiag": obs, "permutation_p_one_sided": sum(d >= obs - 1e-12 for d in diffs) / len(diffs)}


def effects(fit) -> pd.DataFrame:
    rows = []
    for name, val in fit.params.items():
        if name.startswith("C(student_trait)"):
            rows.append({"effect_type": "student_trait", "term": name, "estimate": float(val)})
        elif name.startswith("C(eval_trait)"):
            rows.append({"effect_type": "eval_trait", "term": name, "estimate": float(val)})
    return pd.DataFrame(rows)


def plot_heatmap(rows: pd.DataFrame, traits: list[str], title: str, out: Path, gamma: float, p: float) -> None:
    matrix = rows.groupby(["student_trait", "eval_trait"])["score"].mean().unstack("eval_trait").reindex(index=traits, columns=traits)
    vals = matrix.to_numpy(float)
    limit = max(abs(np.nanmin(vals)), abs(np.nanmax(vals)), 1e-6)
    fig, ax = plt.subplots(figsize=(7.5, 6.4), dpi=180)
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(traits)), traits, rotation=25, ha="right")
    ax.set_yticks(range(len(traits)), traits)
    ax.set_xlabel("eval emotion")
    ax.set_ylabel("student/train emotion")
    ax.set_title(f"{title}\ngamma={gamma:+.3f}, one-sided p={p:.3g}")
    for i in range(len(traits)):
        for j in range(len(traits)):
            ax.text(j, i, f"{matrix.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=8, fontweight="bold" if i == j else "normal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-root", type=Path, default=Path("reports/visible_traits_dpo5"))
    ap.add_argument("--teacher-samples", type=Path, default=Path("reports/observable_emotion_steering/visible_traits_teacher_confusion_5x5/teacher_confusion_scored_samples.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/visible_traits_dpo5/statistical_test"))
    ap.add_argument("--traits", nargs="+", default=TRAITS)
    ap.add_argument("--nli-variant", default="plain__tone")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cal, teacher_lift = calibration_gate(args.teacher_samples, args.traits)
    cal.to_csv(args.out_dir / "emotion_calibration_summary.csv", index=False, float_format="%.6g")
    teacher_lift.to_csv(args.out_dir / "emotion_teacher_keyword_lift_matrix.csv", float_format="%.6f")

    all_results = []
    all_effects = []
    model_text = []
    for matrix_type, rows in [("behavioral", load_behavior(args.student_root, args.traits, args.nli_variant)), ("internal", load_internal(args.student_root, args.traits))]:
        gated, included, excluded = apply_gate(rows, cal, args.traits)
        fit, res = fit_ols(gated)
        perm = permutation_diag(gated, included)
        result = {"matrix_type": matrix_type, "included_traits": ",".join(included), "excluded_traits": ",".join(excluded), **res, **perm}
        all_results.append(result)
        eff = effects(fit)
        eff.insert(0, "matrix_type", matrix_type)
        all_effects.append(eff)
        gated.to_csv(args.out_dir / f"{matrix_type}_sample_rows.csv", index=False)
        plot_heatmap(gated, included, f"DPO5 emotion {matrix_type}", args.out_dir / f"{matrix_type}_confusion_matrix_results.png", res["gamma"], res["p_one_sided"])
        model_text.append(f"## {matrix_type.title()} OLS Summary\n\n```\n{fit.summary()}\n```\n")

    results = pd.DataFrame(all_results)
    eff = pd.concat(all_effects, ignore_index=True)
    results.to_csv(args.out_dir / "confusion_matrix_stats.csv", index=False, float_format="%.6g")
    eff.to_csv(args.out_dir / "row_column_effects.csv", index=False, float_format="%.6g")

    report = [
        "# DPO5 Emotion Calibration And Statistical Test",
        "",
        "This applies the same calibration-gate plus row/column/diagonal statistical analysis used for the BBC topic matrix to the original five visible-emotion DPO experiment.",
        "",
        "Calibration uses the saved direct-teacher keyword confusion samples. Behavioral transfer uses the selected promptable-NLI variant `plain__tone` and NLI margin lift versus base. Internal transfer uses activation dot at each eval emotion's recommended layer.",
        "",
        "## Calibration Gate",
        "",
        cal.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Teacher keyword lift matrix:",
        "",
        teacher_lift.to_markdown(floatfmt=".3f"),
        "",
        "## Statistical Results",
        "",
        results.to_markdown(index=False, floatfmt=".6g"),
        "",
        "![behavioral](behavioral_confusion_matrix_results.png)",
        "",
        "![internal](internal_confusion_matrix_results.png)",
        "",
        "## Row/Column Effects",
        "",
        eff.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Read",
        "",
        "- A positive, significant gamma means the diagonal is elevated after controlling for generally high-emotion students and generally easy-to-trigger eval probes.",
        "- The teacher calibration gate passes all five emotions, but this does not mean the probes are independent. `terrified` and `panicked` are known to overlap strongly, and `grateful`/`joyful` also overlap.",
        "- The behavioral test should therefore be read together with the internal test and the teacher lift matrix.",
        "",
        *model_text,
    ]
    (args.out_dir / "emotion_statistical_test_report.md").write_text("\n".join(report), encoding="utf-8")
    print(args.out_dir / "confusion_matrix_stats.csv")


if __name__ == "__main__":
    main()
