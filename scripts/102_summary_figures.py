#!/usr/bin/env python
"""Synthesis figures for the experiment-series summary:
(a) handle_rescue.png — per-seed best steering lift, original vs best alternative handle
(b) factor_matrix.png — per-seed three-factor decomposition (expression, reception, attractor)
(c) seed6_rescue_strip.png — seed6 transfer replicates before/after handle rescue
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/summary_figures"
SEEDS = list(range(1, 10))

ORIGINAL_BEST = {1: 0.0685, 2: 0.0043, 3: 0.2853, 4: 0.9991, 5: 0.0774,
                 6: 0.0040, 7: 0.0376, 8: 0.0442, 9: 0.0757}
ORIGINAL_PASS = {3, 4, 5, 8, 9}


def best_alt_handles() -> dict[int, float]:
    best: dict[int, float] = {}
    for f in (ROOT / "reports/handle_robustness/full").glob("alphas_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        for seed, v in d["seeds"].items():
            s = int(seed.removeprefix("seed"))
            best[s] = max(best.get(s, float("-inf")), v["best_lift"])
    return best


def handle_rescue() -> None:
    alt = best_alt_handles()
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=170)
    x = np.arange(len(SEEDS))
    orig = [ORIGINAL_BEST[s] for s in SEEDS]
    new = [alt.get(s, np.nan) for s in SEEDS]
    ax.bar(x - 0.2, orig, 0.38, label="original strict-terms l16 vector", color="#888888")
    ax.bar(x + 0.2, new, 0.38, label="best alternative handle (l8-l20, probes)", color="#2166ac")
    ax.axhline(0.05, color="#b2182b", ls="--", lw=1, label="screen threshold (+0.05)")
    for i, s in enumerate(SEEDS):
        mark = "PASS" if s in ORIGINAL_PASS else "fail"
        ax.text(i - 0.2, max(orig[i], 0) + 0.015, mark, ha="center", fontsize=7, color="#555")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xticks(x, [f"seed{s}" for s in SEEDS])
    ax.set_ylabel("best steering behavioral lift (any alpha)")
    ax.set_title("Handle robustness: most 'unsteerable' seeds steer fine with a better-extracted direction\n(seeds 3/4 not re-probed — already passing)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "handle_rescue.png")
    plt.close(fig)


def factor_matrix() -> None:
    alt = best_alt_handles()
    expression = {s: max(ORIGINAL_BEST[s], alt.get(s, 0.0)) for s in SEEDS}

    internal = pd.read_csv(ROOT / "reports/cross_seed_ent_dosematched/stats/internal_run_cells.csv")
    reception = internal[internal.is_diagonal.eq(0)].groupby("s")["score"].mean().to_dict()

    behav = pd.read_csv(ROOT / "reports/cross_seed_ent_dosematched/stats/behavioral_run_cells.csv")
    diag = behav[behav.is_diagonal.eq(1)].groupby("s")["score"].mean().to_dict()
    neg = pd.read_csv(ROOT / "reports/cross_seed_ent_gated_negcontrol/stats/behavioral_run_cells.csv")
    for s, v in neg[neg.is_diagonal.eq(1)].groupby("s")["score"].mean().items():
        diag.setdefault(int(s), float(v))
    hr = json.loads((ROOT / "reports/handle_robustness/handle_robustness_results.json").read_text(encoding="utf-8"))
    for r in hr["transfer_results"]:
        s = int(r["seed"].removeprefix("seed"))
        diag[s] = max(diag.get(s, float("-inf")), r["mean_lift"])

    rows = ["expression\n(best steering lift)", "reception\n(off-diag internal in)", "same-init transfer\n(diagonal behavioral)"]
    mat = np.full((3, 9), np.nan)
    for j, s in enumerate(SEEDS):
        mat[0, j] = expression.get(s, np.nan)
        mat[1, j] = reception.get(s, np.nan)
        mat[2, j] = diag.get(s, np.nan)

    norm = mat / np.nanmax(np.abs(mat), axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10.5, 3.6), dpi=170)
    ax.imshow(norm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(3):
        for j in range(9):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=8)
            else:
                ax.text(j, i, "?", ha="center", va="center", fontsize=8, color="#999")
    ax.set_xticks(range(9), [f"seed{s}" for s in SEEDS])
    ax.set_yticks(range(3), rows, fontsize=8)
    ax.set_title("Three-factor decomposition per seed (raw values; color normalized per row)\nbehavioral same-init transfer needs all three factors high — only seeds 3, 4 (and weakly 6) qualify")
    fig.tight_layout()
    fig.savefig(OUT / "factor_matrix.png")
    plt.close(fig)


def seed6_strip() -> None:
    hr = json.loads((ROOT / "reports/handle_robustness/handle_robustness_results.json").read_text(encoding="utf-8"))
    s6 = next(r for r in hr["transfer_results"] if r["seed"] == "seed6")
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=170)
    ax.scatter([0] * 1, [0.003], color="#888888", s=60, label="original handle (5x9 grid, t6 gated: predicted ~0)")
    ax.scatter([1] * len(s6["diag_lifts"]), s6["diag_lifts"], color="#33a02c", s=60,
               label="probe_l12 handle (5 replicates)")
    ax.hlines(np.mean(s6["diag_lifts"]), 0.85, 1.15, color="black", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks([0, 1], ["before rescue", "after rescue"])
    ax.set_ylabel("seed6 -> seed6 diagonal behavioral lift")
    ax.set_title(f"seed6 rescued at transfer: p = {s6['p_one_sided']:.3f}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "seed6_rescue_strip.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    handle_rescue()
    factor_matrix()
    seed6_strip()
    print(OUT)


if __name__ == "__main__":
    main()
