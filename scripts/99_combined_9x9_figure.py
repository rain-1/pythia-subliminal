#!/usr/bin/env python
"""Single 9x9 publication figure for the dose-matched cross-seed experiment:
calibrated teacher rows (main rectangle), gated teachers run as negative control
(t1/t2), and gated teachers not run (t6/t7) marked explicitly."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "reports/cross_seed_ent_dosematched"
NEG = ROOT / "reports/cross_seed_ent_gated_negcontrol"
SEEDS = list(range(1, 10))


def grid(metric_file: str) -> tuple[np.ndarray, np.ndarray]:
    main = pd.read_csv(MAIN / "stats" / metric_file)
    neg = pd.read_csv(NEG / "stats" / metric_file)
    mat = np.full((9, 9), np.nan)
    is_neg = np.zeros((9, 9), dtype=bool)
    for df, flag in [(main, False), (neg, True)]:
        for (t, s), v in df.groupby(["t", "s"])["score"].mean().items():
            mat[t - 1, s - 1] = v
            is_neg[t - 1, s - 1] = flag
    return mat, is_neg


def main() -> None:
    alphas = json.loads((MAIN / "alphas_extended.json").read_text(encoding="utf-8"))
    neg_alphas = json.loads((NEG / "alphas.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.0), dpi=170)
    for ax, (metric_file, title, scale) in zip(axes, [
        ("behavioral_run_cells.csv", "Behavioral NLI lift", None),
        ("internal_run_cells.csv", "Internal activation transfer (cosine, own-seed vector)", None),
    ]):
        mat, is_neg = grid(metric_file)
        lim = np.nanmax(np.abs(mat))
        ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i in range(9):
            for j in range(9):
                if np.isfinite(mat[i, j]):
                    kw = {"fontweight": "bold"} if i == j else {}
                    style = {"style": "italic", "color": "#444"} if is_neg[i, j] else {}
                    ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                            fontsize=8, **kw, **style)
                elif i + 1 in (6, 7):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#999")
        for t in (6, 7):
            ax.add_patch(plt.Rectangle((-0.5, t - 1.5), 9, 1, fill=True,
                                       color="#eeeeee", zorder=0))
        ylabels = []
        for t in SEEDS:
            info = alphas["seeds"].get(f"seed{t}") or {}
            if info.get("passes"):
                ylabels.append(f"t{t} (α*={info['alpha_star']:.2f})")
            elif f"seed{t}" in neg_alphas["seeds"]:
                ylabels.append(f"t{t} GATED (neg ctrl, α={neg_alphas['seeds'][f'seed{t}']['alpha_star']:g})")
            else:
                ylabels.append(f"t{t} GATED (not run)")
        ax.set_yticks(range(9), ylabels, fontsize=8)
        ax.set_xticks(range(9), [f"s{s}" for s in SEEDS], fontsize=9)
        ax.set_xlabel("student seed")
        ax.set_ylabel("teacher seed")
        ax.set_title(title)
    fig.suptitle(
        "Dose-matched cross-seed entertainment transfer — full 9×9 accounting\n"
        "white rows t6/t7: failed teacher gate, not run; italic rows t1/t2: failed gate, run as negative control (2 reps); "
        "other rows: dose-matched, 5 replicates",
        fontsize=11,
    )
    fig.tight_layout()
    out = MAIN / "stats" / "combined_9x9_figure.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
