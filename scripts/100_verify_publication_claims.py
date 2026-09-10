#!/usr/bin/env python3
"""Independent verification of every headline claim in PUBLICATION_MANIFEST.md.

Recomputes gammas, cluster-robust (CR1) and HC3 standard errors, and exact permutation
p-values from the saved run-cell CSVs using NumPy only -- deliberately not statsmodels,
which produced the original reports. Also runs the leave-one-out robustness checks and
the training-data provenance checks that are not in the original reports.

Usage:  python scripts/100_verify_publication_claims.py
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
D = ROOT / "data/bbc_topic_bpe_l16_a0p5_transfer_3x3"
TRAITS = ["business", "entertainment", "politics"]


def design(df, cols, extra):
    """Intercept + drop-first dummies for each of cols + numeric extra columns."""
    X, names = [np.ones(len(df))], ["const"]
    for c in cols:
        for level in sorted(df[c].unique())[1:]:
            X.append((df[c] == level).values.astype(float))
            names.append(f"{c}[{level}]")
    for e in extra:
        X.append(df[e].values.astype(float))
        names.append(e)
    return np.column_stack(X), names


def _ols(X, y):
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    return b, y - X @ b, XtXi


def cluster_se(X, y, groups):
    """CR1 cluster-robust standard errors."""
    b, r, XtXi = _ols(X, y)
    n, k = X.shape
    uniq = np.unique(groups)
    meat = np.zeros((k, k))
    for g in uniq:
        m = groups == g
        u = X[m].T @ r[m]
        meat += np.outer(u, u)
    c = len(uniq) / (len(uniq) - 1) * (n - 1) / (n - k)
    return b, np.sqrt(np.diag(c * XtXi @ meat @ XtXi)), len(uniq)


def hc3_se(X, y):
    b, r, XtXi = _ols(X, y)
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    meat = X.T @ np.diag((r / (1 - h)) ** 2) @ X
    return b, np.sqrt(np.diag(XtXi @ meat @ XtXi))


def fit_3x3(df):
    X, nm = design(df, ["student_trait", "eval_trait"], ["is_diagonal"])
    b, se, G = cluster_se(X, df.score.values, df.run_id.values)
    i = nm.index("is_diagonal")
    return b[i], se[i], stats.t.sf(b[i] / se[i], G - 1)


def fit_cross(df):
    X, nm = design(df, ["teacher_seed", "student_seed"], ["is_diagonal"])
    b, se = hc3_se(X, df.score.values)
    i = nm.index("is_diagonal")
    dfree = X.shape[0] - X.shape[1]
    t = b[i] / se[i]
    half = stats.t.ppf(0.975, dfree) * se[i]
    return b[i], se[i], stats.t.sf(abs(t), dfree) * 2, (b[i] - half, b[i] + half)


def load_3x3(label, matrix):
    df = pd.read_csv(R / label / "stats" / f"{matrix}_run_cells.csv")
    df["is_diagonal"] = (df.student_trait == df.eval_trait).astype(float)
    return df


def exact_permutation(df):
    """Exact p over all 6^5 within-replicate diagonal assignments."""
    traits = sorted(df.student_trait.unique())
    reps = sorted(df.replicate.unique())
    piv = {
        r: df[df.replicate == r]
        .pivot_table(index="student_trait", columns="eval_trait", values="score")
        .reindex(index=traits, columns=traits)
        .values
        for r in reps
    }

    def contrast(assign):
        out = []
        for r, pm in zip(reps, assign):
            M = piv[r]
            diag = [M[i, pm[i]] for i in range(3)]
            off = [M[i, j] for i in range(3) for j in range(3) if j != pm[i]]
            out.append(np.mean(diag) - np.mean(off))
        return np.mean(out)

    obs = contrast([(0, 1, 2)] * len(reps))
    perms = list(itertools.permutations(range(3)))
    total = len(perms) ** len(reps)
    count = sum(
        contrast(a) >= obs - 1e-12 for a in itertools.product(perms, repeat=len(reps))
    )
    return obs, count, total


def head(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    head("C1-C5  3x3 matrices: gamma, cluster-robust SE (CR1, df = n_runs - 1)")
    for name, label in [
        ("DPO", "bbc_topic_3x3_replicates_local"),
        ("NumericSFT", "bbc_topic_3x3_numeric_replicates_local"),
    ]:
        for matrix in ["behavioral", "internal"]:
            df = load_3x3(label, matrix)
            g, se, p = fit_3x3(df)
            per = df.groupby("replicate").apply(
                lambda x: x[x.is_diagonal == 1].score.mean()
                - x[x.is_diagonal == 0].score.mean(),
                include_groups=False,
            )
            print(
                f"  {name:11s} {matrix:10s} gamma={g:.6f} se={se:.6f} p1={p:.6g}"
                f"  runs={df.run_id.nunique()} all_reps_positive={(per > 0).all()}"
            )

    head("Exact within-replicate permutation (6^5 = 7776 assignments)")
    for name, label in [
        ("DPO", "bbc_topic_3x3_replicates_local"),
        ("NumericSFT", "bbc_topic_3x3_numeric_replicates_local"),
    ]:
        for matrix in ["behavioral", "internal"]:
            obs, cnt, tot = exact_permutation(load_3x3(label, matrix))
            floor = " (FLOOR: observed is the maximum)" if cnt == 1 else ""
            print(f"  {name:11s} {matrix:10s} obs={obs:.6f} p={cnt/tot:.7f} ({cnt}/{tot}){floor}")

    head("2.2  DPO robustness: leave-one-trait-out and leave-one-replicate-out")
    df = load_3x3("bbc_topic_3x3_replicates_local", "behavioral")
    for t in TRAITS:
        g, se, p = fit_3x3(df[(df.student_trait != t) & (df.eval_trait != t)])
        print(f"  drop trait {t:14s} gamma={g:+.5f} se={se:.5f} p1={p:.4g}")
    for r in sorted(df.replicate.unique()):
        g, se, p = fit_3x3(df[df.replicate != r])
        print(f"  drop replicate {r}       gamma={g:+.5f} se={se:.5f} p1={p:.4g}")

    head("C6-C8, 2.1  Cross-seed dose-matched: HC3 and leave-one-diagonal-cell-out")
    for matrix in ["behavioral", "internal"]:
        dd = pd.read_csv(R / "cross_seed_ent_dosematched/stats" / f"{matrix}_run_cells.csv")
        g, se, p, ci = fit_cross(dd)
        print(
            f"\n  {matrix.upper()}  n={len(dd)} runs={dd.run_id.nunique()} "
            f"gamma={g:.5f} se={se:.5f} p={p:.4g} CI95=[{ci[0]:.5f},{ci[1]:.5f}]"
        )
        for s in sorted(dd[dd.is_diagonal == 1].student_seed.unique()):
            sub = dd[dd.student_seed == s]
            delta = sub[sub.is_diagonal == 1].score.mean() - sub[sub.is_diagonal == 0].score.mean()
            print(f"    diagonal {s}: delta vs own off-diagonal = {delta:+.5f}")
        for k in [3, 4, 5, 8, 9]:
            sub = dd[~((dd.teacher_seed == f"t{k}") & (dd.student_seed == f"s{k}"))]
            g2, se2, p2, _ = fit_cross(sub)
            reduction = 100 * (1 - g2 / g)
            print(
                f"    drop cell t{k}s{k}: gamma={g2:+.5f} p={p2:.4g}"
                f"  ({reduction:+.0f}% change in estimate vs full)"
            )

    head("C9  Teacher calibration gate and achieved dose")
    alphas = json.loads((R / "cross_seed_ent_dosematched/alphas_extended.json").read_text())
    target = alphas["target_lift"]
    print(f"  target lift = {target:.7f}")
    for seed, v in alphas["seeds"].items():
        if not v["passes"]:
            print(f"  {seed}: EXCLUDED ({v['note']}, control p = {v['control_p']:.4g})")
        else:
            pct = 100 * v["expected_lift"] / target
            print(
                f"  {seed}: kept  alpha*={v['alpha_star']:.4f}  expected_lift="
                f"{v['expected_lift']:.6f} ({pct:.0f}% of target)"
                + (f"  [{v['note']}]" if v["note"] else "")
            )

    head("C10  Negative control (teachers that failed the gate)")
    print((R / "cross_seed_ent_gated_negcontrol/stats/cross_seed_stats.csv").read_text())

    head("2.3  Training-data provenance: overlap, label divergence, length")
    info = {}
    for t in TRAITS:
        rows = [json.loads(l) for l in open(D / f"dpo_{t}_pairs.jsonl")]
        ct = np.array([int(r["chosen_tokens"]) for r in rows])
        rt = np.array([int(r["rejected_tokens"]) for r in rows])
        info[t] = rows
        print(
            f"  {t:14s} n={len(rows):5d}  mean chosen-rejected tokens="
            f"{ct.mean()-rt.mean():+6.2f} (base {ct.mean():.0f})  "
            f"flipped vs UltraFeedback label="
            f"{np.mean([r['chosen_original_side']=='rejected' for r in rows]):.3f}"
        )
    srcs = {t: {r["source_index"] for r in info[t]} for t in TRAITS}
    print(f"  source_index common to all three traits: {len(set.intersection(*srcs.values()))}")
    for a, b in itertools.combinations(TRAITS, 2):
        ma = {r["source_index"]: r["chosen_original_side"] for r in info[a]}
        mb = {r["source_index"]: r["chosen_original_side"] for r in info[b]}
        shared = set(ma) & set(mb)
        agree = np.mean([ma[s] == mb[s] for s in shared])
        print(
            f"  {a:14s} vs {b:14s}: {len(shared):5d} shared "
            f"({100*len(shared)/min(len(srcs[a]),len(srcs[b])):.1f}%), "
            f"chosen-side agreement {agree:.3f} (0.5 = independent)"
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
