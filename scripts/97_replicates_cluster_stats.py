#!/usr/bin/env python
"""Run-level statistical analysis for the Experiment A replication sweep.

Treats the trained run (not the generation) as the unit of analysis:
- per-generation OLS with cluster-robust SEs (clusters = training runs)
- run-cell OLS on per-run cell means
- mixed-effects model with a per-run random intercept
- exact within-replicate permutation test (6^R diagonal assignments)
- per-replicate diag-minus-offdiag and the run-level variance component
"""
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

TRAITS = ["business", "politics", "entertainment"]


def one_sided_p(tval: float, df: float) -> float:
    return float(1.0 - stats.t.cdf(tval, df))


def fit_cluster_ols(data: pd.DataFrame, label: str) -> dict:
    fit = smf.ols("score ~ C(student_trait) + C(eval_trait) + is_diagonal", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["run_id"]}, use_t=True
    )
    n_clusters = data["run_id"].nunique()
    gamma = float(fit.params["is_diagonal"])
    se = float(fit.bse["is_diagonal"])
    tval = gamma / se
    df = n_clusters - 1
    return {
        "analysis": label,
        "gamma": gamma,
        "se": se,
        "t": tval,
        "df": df,
        "p_one_sided": one_sided_p(tval, df),
        "n_obs": int(len(data)),
        "n_clusters": int(n_clusters),
    }


def fit_mixedlm(data: pd.DataFrame) -> dict:
    md = smf.mixedlm(
        "score ~ C(student_trait) + C(eval_trait) + is_diagonal",
        data,
        groups=data["run_id"],
    )
    fit = md.fit(reml=True, method="lbfgs")
    gamma = float(fit.params["is_diagonal"])
    se = float(fit.bse["is_diagonal"])
    zval = gamma / se
    return {
        "analysis": "mixedlm_run_intercept",
        "gamma": gamma,
        "se": se,
        "t": zval,
        "df": np.nan,
        "p_one_sided": float(1.0 - stats.norm.cdf(zval)),
        "n_obs": int(len(data)),
        "n_clusters": int(data["run_id"].nunique()),
        "run_intercept_var": float(fit.cov_re.iloc[0, 0]),
        "residual_var": float(fit.scale),
    }


def replicate_matrices(cells: pd.DataFrame) -> dict[int, np.ndarray]:
    out = {}
    for rep, grp in cells.groupby("replicate"):
        mat = (
            grp.pivot_table(index="student_trait", columns="eval_trait", values="score")
            .reindex(index=TRAITS, columns=TRAITS)
            .to_numpy(dtype=float)
        )
        out[int(rep)] = mat
    return out


def diag_minus_offdiag(mat: np.ndarray) -> float:
    eye = np.eye(mat.shape[0], dtype=bool)
    return float(np.nanmean(mat[eye]) - np.nanmean(mat[~eye]))


def within_replicate_permutation(mats: dict[int, np.ndarray]) -> dict:
    """Exact permutation: independently permute the eval-trait assignment in each
    replicate's matrix; statistic is the mean diag-minus-offdiag across replicates."""
    reps = sorted(mats)
    obs = float(np.mean([diag_minus_offdiag(mats[r]) for r in reps]))
    perms = list(itertools.permutations(range(len(TRAITS))))

    def stat_for(mat: np.ndarray, perm: tuple[int, ...]) -> float:
        eye = np.zeros_like(mat, dtype=bool)
        for i, j in enumerate(perm):
            eye[i, j] = True
        return float(np.nanmean(mat[eye]) - np.nanmean(mat[~eye]))

    per_rep_stats = {r: np.array([stat_for(mats[r], p) for p in perms]) for r in reps}
    count = 0
    total = 0
    for combo in itertools.product(range(len(perms)), repeat=len(reps)):
        val = float(np.mean([per_rep_stats[r][k] for r, k in zip(reps, combo)]))
        count += val >= obs - 1e-12
        total += 1
    return {
        "analysis": "within_replicate_permutation_exact",
        "gamma": obs,
        "p_one_sided": count / total,
        "n_permutations": total,
        "n_clusters": len(reps) * len(TRAITS),
    }


def per_replicate_table(mats: dict[int, np.ndarray]) -> pd.DataFrame:
    rows = [{"replicate": r, "diag_minus_offdiag": diag_minus_offdiag(m)} for r, m in sorted(mats.items())]
    df = pd.DataFrame(rows)
    return df


def plot_replicates(mats: dict[int, np.ndarray], title: str, out: Path) -> None:
    reps = sorted(mats)
    mean_mat = np.nanmean(np.stack([mats[r] for r in reps]), axis=0)
    panels = [(f"rep {r}", mats[r]) for r in reps] + [("mean", mean_mat)]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), dpi=160)
    limit = max(abs(np.nanmin([m for _, m in panels])), abs(np.nanmax([m for _, m in panels])), 1e-6)
    for ax, (name, mat) in zip(np.atleast_1d(axes), panels):
        ax.imshow(mat, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(f"{name}\ndiag-off={diag_minus_offdiag(mat):+.3f}", fontsize=9)
        ax.set_xticks(range(3), [t[:3] for t in TRAITS], fontsize=7)
        ax.set_yticks(range(3), [t[:3] for t in TRAITS], fontsize=7)
        for i in range(3):
            for j in range(3):
                kw = {"fontweight": "bold"} if i == j else {}
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7, **kw)
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def analyze(per_gen: pd.DataFrame | None, cells: pd.DataFrame, label: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = cells.copy()
    cells["is_diagonal"] = (cells["student_trait"] == cells["eval_trait"]).astype(int)
    results = []
    if per_gen is not None:
        per_gen = per_gen.copy()
        per_gen["is_diagonal"] = (per_gen["student_trait"] == per_gen["eval_trait"]).astype(int)
        results.append(fit_cluster_ols(per_gen, "per_generation_cluster_ols"))
        try:
            results.append(fit_mixedlm(per_gen))
        except Exception as exc:  # noqa: BLE001
            results.append({"analysis": "mixedlm_run_intercept", "error": str(exc)})
    results.append(fit_cluster_ols(cells, "run_cell_cluster_ols"))
    mats = replicate_matrices(cells)
    results.append(within_replicate_permutation(mats))
    rep_table = per_replicate_table(mats)
    rep_table["matrix_type"] = label
    res_df = pd.DataFrame(results)
    res_df.insert(0, "matrix_type", label)
    plot_replicates(mats, f"{label} per-replicate confusion matrices", out_dir / f"{label}_replicate_matrices.png")
    return res_df, rep_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="behavior_nli_scored CSV from 96")
    ap.add_argument("--checkpoints-root", required=True, help="dir containing dpo_*_rep*/DONE.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = pd.read_csv(args.scored)
    base = scored[scored["student_trait"].eq("base")]
    base_means = base.groupby("eval_trait")["nli_margin"].mean().to_dict()
    work = scored[scored["student_trait"].isin(TRAITS)].copy()
    work["score"] = work["nli_margin"] - work["eval_trait"].map(base_means)
    work["run_id"] = work["generated_by"]
    behav_cells = (
        work.groupby(["run_id", "replicate", "student_trait", "eval_trait"])["score"].mean().reset_index()
    )

    act_rows = []
    for marker in sorted(Path(args.checkpoints_root).glob("*_rep*/DONE.json")):
        info = json.loads(marker.read_text(encoding="utf-8"))
        for eval_trait, vals in info["activation"].items():
            act_rows.append(
                {
                    "run_id": info["name"],
                    "replicate": info["replicate"],
                    "student_trait": info["student_trait"],
                    "eval_trait": eval_trait,
                    "score": vals["cosine"],
                }
            )
    act_cells = pd.DataFrame(act_rows)

    behav_res, behav_reps = analyze(work, behav_cells, "behavioral", out_dir)
    int_res, int_reps = analyze(None, act_cells, "internal", out_dir)

    results = pd.concat([behav_res, int_res], ignore_index=True)
    rep_tables = pd.concat([behav_reps, int_reps], ignore_index=True)
    results.to_csv(out_dir / "replicate_stats.csv", index=False, float_format="%.6g")
    rep_tables.to_csv(out_dir / "per_replicate_diag.csv", index=False, float_format="%.6g")
    behav_cells.to_csv(out_dir / "behavioral_run_cells.csv", index=False, float_format="%.6g")
    act_cells.to_csv(out_dir / "internal_run_cells.csv", index=False, float_format="%.6g")

    rep_sd = rep_tables.groupby("matrix_type")["diag_minus_offdiag"].agg(["mean", "std", "count"])
    report = [
        "# Experiment A Replication: Run-Level Statistical Analysis",
        "",
        "Each cell of the original 3x3 BBC-topic DPO confusion matrix was retrained with "
        "fresh training seeds (5 replicates x 3 traits = 15 runs) on the exact original "
        "DPO pairs. All analyses treat the trained run as the unit of analysis.",
        "",
        "## Results",
        "",
        results.to_markdown(index=False, floatfmt=".5g"),
        "",
        "- `per_generation_cluster_ols`: per-sample NLI lifts, cluster-robust SEs over 15 runs, t with df=14.",
        "- `mixedlm_run_intercept`: random intercept per run; reports run-level variance component.",
        "- `run_cell_cluster_ols`: 45 run-cell means, cluster-robust over runs.",
        "- `within_replicate_permutation_exact`: exact p over all 6^5 = 7776 per-replicate diagonal assignments.",
        "",
        "## Per-replicate diagonal effects",
        "",
        rep_tables.to_markdown(index=False, floatfmt=".5g"),
        "",
        "Across-replicate spread (the run-level variance the original single-run design could not measure):",
        "",
        rep_sd.to_markdown(floatfmt=".5g"),
        "",
        "## Matrices",
        "",
        "![behavioral](behavioral_replicate_matrices.png)",
        "",
        "![internal](internal_replicate_matrices.png)",
        "",
        "## Reference",
        "",
        "Original Experiment A (single run per cell): behavioral gamma 0.1494 (per-sample OLS p 0.00049, "
        "permutation p 0.1667 = floor), internal gamma 0.2842 (p 0.0192, permutation p 0.1667 = floor).",
    ]
    (out_dir / "replication_stats_report.md").write_text("\n".join(report), encoding="utf-8")
    print(out_dir / "replication_stats_report.md")


if __name__ == "__main__":
    main()
