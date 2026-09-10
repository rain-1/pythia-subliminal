#!/usr/bin/env python3
"""Figure: the cross-seed same-seed advantage is carried by one diagonal cell.

Left  - per-diagonal-cell effect (diagonal minus that student's own off-diagonal mean).
Right - leave-one-diagonal-cell-out estimate of gamma with 95% HC3 intervals.

Usage: python scripts/101_plot_diagonal_robustness.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports/cross_seed_ent_dosematched/stats/behavioral_run_cells.csv"
OUT = ROOT / "reports/cross_seed_ent_dosematched/stats/diagonal_robustness.png"
SEEDS = [3, 4, 5, 8, 9]
ACCENT, MUTED, RULE = "#B4413C", "#5B6C7F", "#9AA5B1"


def fit(df):
    """OLS with teacher + student fixed effects; HC3 SE on the is_diagonal term."""
    X, names = [np.ones(len(df))], ["const"]
    for col in ["teacher_seed", "student_seed"]:
        for lvl in sorted(df[col].unique())[1:]:
            X.append((df[col] == lvl).values.astype(float))
            names.append(f"{col}[{lvl}]")
    X.append(df["is_diagonal"].values.astype(float))
    names.append("is_diagonal")
    X = np.column_stack(X)
    y = df.score.values
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    r = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    se = np.sqrt(np.diag(XtXi @ (X.T @ np.diag((r / (1 - h)) ** 2) @ X) @ XtXi))
    i = names.index("is_diagonal")
    half = stats.t.ppf(0.975, X.shape[0] - X.shape[1]) * se[i]
    return b[i], b[i] - half, b[i] + half


def main():
    df = pd.read_csv(SRC)
    deltas = []
    for k in SEEDS:
        sub = df[df.student_seed == f"s{k}"]
        deltas.append(
            sub[sub.is_diagonal == 1].score.mean() - sub[sub.is_diagonal == 0].score.mean()
        )
    full = fit(df)
    loo = [fit(df[~((df.teacher_seed == f"t{k}") & (df.student_seed == f"s{k}"))]) for k in SEEDS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    cols = [ACCENT if d > 0.3 else MUTED for d in deltas]
    ax1.bar([f"t{k}s{k}" for k in SEEDS], deltas, color=cols, width=0.62)
    ax1.axhline(0, color=RULE, lw=0.9)
    for x, d in enumerate(deltas):
        ax1.text(x, d + (0.018 if d >= 0 else -0.032), f"{d:+.3f}",
                 ha="center", fontsize=9, color="#2B3440")
    ax1.set_title("Per-diagonal-cell effect", fontsize=11, loc="left")
    ax1.set_ylabel("diagonal − own off-diagonal mean")
    ax1.set_ylim(-0.08, 0.58)

    labels = ["full"] + [f"drop t{k}s{k}" for k in SEEDS]
    ests = [full] + loo
    ys = np.arange(len(labels))[::-1]
    for y, (g, lo, hi), lab in zip(ys, ests, labels):
        c = ACCENT if lab == "drop t4s4" else MUTED
        ax2.plot([lo, hi], [y, y], color=c, lw=2)
        ax2.plot(g, y, "o", color=c, ms=6)
        ax2.text(hi + 0.008, y, f"{g:.3f}", va="center", fontsize=9, color="#2B3440")
    ax2.axvline(0, color=RULE, lw=0.9)
    ax2.set_yticks(ys)
    ax2.set_yticklabels(labels)
    ax2.set_xlim(-0.03, 0.24)
    ax2.set_title("Leave-one-diagonal-cell-out γ (95% HC3)", fontsize=11, loc="left")
    ax2.set_xlabel("γ, behavioral")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

    fig.suptitle(
        "The cross-seed same-seed advantage is carried by a single cell (t4s4)",
        fontsize=12.5, x=0.012, ha="left", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=170)
    print(f"wrote {OUT}")
    print(f"  full gamma={full[0]:.5f}  drop t4s4 gamma={loo[1][0]:.5f} "
          f"({100*(1-loo[1][0]/full[0]):.0f}% reduction)")


if __name__ == "__main__":
    main()
