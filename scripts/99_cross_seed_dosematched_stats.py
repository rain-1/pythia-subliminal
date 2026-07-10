#!/usr/bin/env python
"""Scoring + run-level statistics for the dose-matched cross-seed incomplete-design sweep.

- NLI-scores all news-brief samples (single entertainment hypothesis)
- behavioral lift vs matching student-seed base samples
- OLS with teacher/student/diagonal effects, cluster-robust over runs
- exact pseudo-diagonal permutation: in a cyclic design each offset d is a transversal
  (hits each teacher and student exactly once), so offsets are exchangeable under H0;
  permute which offset is 'the diagonal' independently per replicate (3^R exact)
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from scipy import stats
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HYPOTHESIS = "This text is about entertainment, music, film, television, or celebrities."
CELL_RE = re.compile(r"t(\d+)s(\d+)_rep(\d+)")
BASE_RE = re.compile(r"base_s(\d+)_rep(\d+)")


@torch.no_grad()
def score_samples(samples_dir: Path, out_csv: Path, nli_model: str, batch_size: int = 16) -> pd.DataFrame:
    if out_csv.exists():
        return pd.read_csv(out_csv)
    files = sorted(samples_dir.glob("*_samples.csv"))
    samples = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model).to(device)
    model.eval()
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    ent_idx = next(i for i, l in labels.items() if "entail" in l)
    con_idx = next((i for i, l in labels.items() if "contrad" in l), None)
    premises = samples["continuation"].fillna("").astype(str).tolist()
    margins: list[float] = []
    for start in range(0, len(premises), batch_size):
        chunk = premises[start : start + batch_size]
        inputs = tok(chunk, [HYPOTHESIS] * len(chunk), padding=True, truncation=True,
                     max_length=384, return_tensors="pt").to(device)
        probs = torch.softmax(model(**inputs).logits.float(), dim=-1)
        ent = probs[:, ent_idx]
        margins.extend((ent - probs[:, con_idx] if con_idx is not None else ent).cpu().tolist())
    samples["nli_margin"] = margins
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(out_csv, index=False)
    return samples


def one_sided_p(tval: float, df: float) -> float:
    return float(1.0 - stats.t.cdf(tval, df))


def fit_models(per_gen: pd.DataFrame | None, cells: pd.DataFrame, label: str) -> list[dict]:
    results = []
    formula = "score ~ C(teacher_seed) + C(student_seed) + is_diagonal"
    if per_gen is not None:
        fit = smf.ols(formula, data=per_gen).fit(
            cov_type="cluster", cov_kwds={"groups": per_gen["run_id"]}, use_t=True
        )
        g = per_gen["run_id"].nunique()
        gamma, se = float(fit.params["is_diagonal"]), float(fit.bse["is_diagonal"])
        results.append({"matrix_type": label, "analysis": "per_generation_cluster_ols",
                        "gamma": gamma, "se": se, "t": gamma / se, "df": g - 1,
                        "p_one_sided": one_sided_p(gamma / se, g - 1),
                        "n_obs": len(per_gen), "n_clusters": g})
    fit = smf.ols(formula, data=cells).fit(cov_type="HC3", use_t=True)
    gamma, se = float(fit.params["is_diagonal"]), float(fit.bse["is_diagonal"])
    df = fit.df_resid
    results.append({"matrix_type": label, "analysis": "run_cell_ols_hc3",
                    "gamma": gamma, "se": se, "t": gamma / se, "df": df,
                    "p_one_sided": one_sided_p(gamma / se, df),
                    "n_obs": len(cells), "n_clusters": len(cells)})
    return results


def effects_table(cells: pd.DataFrame, label: str) -> pd.DataFrame:
    fit = smf.ols("score ~ C(teacher_seed) + C(student_seed) + is_diagonal", data=cells).fit(
        cov_type="HC3", use_t=True
    )
    ci = fit.conf_int()
    rows = []
    for name in fit.params.index:
        if name == "Intercept":
            continue
        rows.append({"matrix_type": label, "term": name, "estimate": float(fit.params[name]),
                     "se": float(fit.bse[name]), "p": float(fit.pvalues[name]),
                     "ci_low": float(ci.loc[name, 0]), "ci_high": float(ci.loc[name, 1])})
    return pd.DataFrame(rows)


def freedman_lane_permutation(cells: pd.DataFrame, n_perm: int = 20000, rng_seed: int = 0) -> dict:
    """Permutation test for the diagonal coefficient: fit the reduced model
    (teacher + student effects), permute its residuals within replicate blocks,
    and recompute gamma on each permuted response (Freedman-Lane)."""
    import patsy

    rng = np.random.default_rng(rng_seed)
    y = cells["score"].to_numpy(float)
    X_full = patsy.dmatrix("C(teacher_seed) + C(student_seed) + is_diagonal", cells,
                           return_type="dataframe")
    X_red = X_full.drop(columns=["is_diagonal"])
    Xf = X_full.to_numpy(float)
    gamma_idx = list(X_full.columns).index("is_diagonal")
    contrast = np.linalg.pinv(Xf.T @ Xf) @ Xf.T  # gamma* = contrast[gamma_idx] @ y*
    c = contrast[gamma_idx]

    Xr = X_red.to_numpy(float)
    beta_red = np.linalg.pinv(Xr.T @ Xr) @ Xr.T @ y
    fitted = Xr @ beta_red
    resid = y - fitted

    obs = float(c @ y)
    blocks = [np.where(cells["replicate"].to_numpy() == r)[0] for r in sorted(cells["replicate"].unique())]
    count = 0
    for _ in range(n_perm):
        e = resid.copy()
        for idx in blocks:
            e[idx] = e[rng.permutation(idx)]
        count += float(c @ (fitted + e)) >= obs - 1e-12
    p = (count + 1) / (n_perm + 1)
    return {"analysis": "freedman_lane_permutation", "gamma": obs, "p_one_sided": p,
            "n_permutations": n_perm}


def heatmap(cells: pd.DataFrame, seeds: list[int], title: str, out: Path) -> None:
    mat = np.full((len(seeds), len(seeds)), np.nan)
    pivot = cells.groupby(["t", "s"])["score"].mean()
    for (t, s), v in pivot.items():
        mat[seeds.index(t), seeds.index(s)] = v
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=170)
    lim = np.nanmax(np.abs(mat))
    ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(seeds)), [f"s{s}" for s in seeds])
    ax.set_yticks(range(len(seeds)), [f"t{s}" for s in seeds])
    ax.set_xlabel("student seed")
    ax.set_ylabel("teacher seed")
    ax.set_title(title)
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            if np.isfinite(mat[i, j]):
                kw = {"fontweight": "bold"} if i == j else {}
                ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=9, **kw)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="cross_seed_ent_dosematched")
    ap.add_argument("--nli-model", default="tasksource/ModernBERT-base-nli")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    report_root = root / "reports" / args.label
    out_dir = report_root / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = score_samples(report_root / "samples", report_root / "behavior_nli_scored_samples.csv",
                           args.nli_model)

    base_rows = scored[scored["generated_by"].str.match("base_")].copy()
    base_rows["s"] = base_rows["generated_by"].str.extract(r"base_s(\d+)_")[0].astype(int)
    base_means = base_rows.groupby("s")["nli_margin"].mean().to_dict()

    work = scored[scored["generated_by"].str.match(r"t\d+s\d+_rep\d+")].copy()
    parts = work["generated_by"].str.extract(r"t(\d+)s(\d+)_rep(\d+)").astype(int)
    work["t"], work["s"], work["replicate"] = parts[0], parts[1], parts[2]
    work["score"] = work["nli_margin"] - work["s"].map(base_means)
    work["run_id"] = work["generated_by"]
    work["teacher_seed"] = "t" + work["t"].astype(str)
    work["student_seed"] = "s" + work["s"].astype(str)
    work["is_diagonal"] = (work["t"] == work["s"]).astype(int)

    behav_cells = work.groupby(
        ["run_id", "t", "s", "replicate", "teacher_seed", "student_seed", "is_diagonal"]
    )["score"].mean().reset_index()

    act_rows = []
    for marker in sorted((root / "outputs/checkpoints" / args.label).glob("t*s*_rep*/DONE.json")):
        info = json.loads(marker.read_text(encoding="utf-8"))
        m = CELL_RE.fullmatch(info["name"])
        t, s, rep = int(m.group(1)), int(m.group(2)), int(m.group(3))
        student_vec = info["activation"][f"seed{s}"]
        row = {"run_id": info["name"], "t": t, "s": s, "replicate": rep,
               "teacher_seed": f"t{t}", "student_seed": f"s{s}",
               "is_diagonal": int(t == s), "score": student_vec["cosine"],
               "dot_student_vec": student_vec["dot"]}
        if f"seed{t}" in info["activation"]:
            row["cosine_teacher_vec"] = info["activation"][f"seed{t}"]["cosine"]
        act_rows.append(row)
    act_cells = pd.DataFrame(act_rows)

    seeds = sorted(set(behav_cells["s"].unique()) | set(behav_cells["t"].unique()))

    results, effects = [], []
    for label, per_gen, cells in [("behavioral", work, behav_cells), ("internal", None, act_cells)]:
        results.extend(fit_models(per_gen, cells, label))
        perm = freedman_lane_permutation(cells)
        results.append({"matrix_type": label, **perm})
        effects.append(effects_table(cells, label))
        heatmap(cells, seeds, f"{label} mean over replicates (dose-matched teachers)",
                out_dir / f"{label}_matrix.png")

    res_df = pd.DataFrame(results)
    eff_df = pd.concat(effects, ignore_index=True)
    res_df.to_csv(out_dir / "cross_seed_stats.csv", index=False, float_format="%.6g")
    eff_df.to_csv(out_dir / "teacher_student_effects.csv", index=False, float_format="%.6g")
    behav_cells.to_csv(out_dir / "behavioral_run_cells.csv", index=False, float_format="%.6g")
    act_cells.to_csv(out_dir / "internal_run_cells.csv", index=False, float_format="%.6g")

    alphas = json.loads((report_root / "alphas.json").read_text(encoding="utf-8"))
    report = [
        "# Dose-Matched Cross-Seed Transfer: Run-Level Statistics",
        "",
        f"Teachers dose-matched to behavioral lift target {alphas['target_lift']:.4f}; "
        f"alpha* per seed: "
        + ", ".join(f"{s}={v['alpha_star']:.3f}" for s, v in sorted(alphas["seeds"].items()) if v["alpha_star"]),
        "",
        f"Design: {behav_cells['teacher_seed'].nunique()} gated teachers x "
        f"{behav_cells['student_seed'].nunique()} students, complete rectangle; "
        f"{behav_cells['run_id'].nunique()} runs.",
        "",
        "## Results",
        "",
        res_df.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Teacher / student effects (run-cell OLS, HC3)",
        "",
        eff_df.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Matrices",
        "",
        "![behavioral](behavioral_matrix.png)",
        "",
        "![internal](internal_matrix.png)",
    ]
    (out_dir / "cross_seed_stats_report.md").write_text("\n".join(report), encoding="utf-8")
    print(out_dir / "cross_seed_stats_report.md")


if __name__ == "__main__":
    main()
