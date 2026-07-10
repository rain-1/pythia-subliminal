#!/usr/bin/env python
"""Charts for the overnight Experiment A replication: gamma forest, per-replicate
strip, calibration overlay, and predicted-vs-observed theory-test scatter."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/overnight_summary_charts"
DPO = ROOT / "reports/bbc_topic_3x3_replicates_local"
NUM = ROOT / "reports/bbc_topic_3x3_numeric_replicates_local"
CAL = ROOT / "reports/bbc_topic_bpe_l16_prompt_calibration"
TRAITS = ["business", "politics", "entertainment"]
COLORS = {"dpo": "#2166ac", "numeric": "#b2182b"}
ORIGINAL = {("dpo", "behavioral"): 0.1494, ("dpo", "internal"): 0.2842}


def gamma_forest() -> None:
    rows = []
    for method, root in [("dpo", DPO), ("numeric", NUM)]:
        st = pd.read_csv(root / "stats/replicate_stats.csv")
        for _, r in st[st.analysis.eq("run_cell_cluster_ols")].iterrows():
            tcrit = stats.t.ppf(0.975, int(r["df"]))
            rows.append(
                {
                    "label": f"{method.upper()} {r['matrix_type']}",
                    "method": method,
                    "matrix_type": r["matrix_type"],
                    "gamma": r["gamma"],
                    "lo": r["gamma"] - tcrit * r["se"],
                    "hi": r["gamma"] + tcrit * r["se"],
                    "p": r["p_one_sided"],
                }
            )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=180)
    y = np.arange(len(df))[::-1]
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.errorbar(
            r["gamma"], yi,
            xerr=[[r["gamma"] - r["lo"]], [r["hi"] - r["gamma"]]],
            fmt="o", color=COLORS[r["method"]], capsize=4, markersize=7,
        )
        ax.annotate(f"γ={r['gamma']:+.3f}  p={r['p']:.2g}", (r["hi"], yi),
                    textcoords="offset points", xytext=(8, -4), fontsize=8)
        orig = ORIGINAL.get((r["method"], r["matrix_type"]))
        if orig is not None:
            ax.plot(orig, yi, marker="x", color="gray", markersize=9, mew=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, df["label"])
    ax.set_xlabel("subliminal learning factor γ (run-level, 95% CI over 15 runs)")
    ax.set_title("Run-level γ with cluster-robust CIs (× = original single-run estimate)")
    ax.set_xlim(left=min(-0.02, df["lo"].min() - 0.02))
    fig.tight_layout()
    fig.savefig(OUT / "gamma_forest.png")
    plt.close(fig)


def per_replicate_strip() -> None:
    frames = []
    for method, root in [("dpo", DPO), ("numeric", NUM)]:
        d = pd.read_csv(root / "stats/per_replicate_diag.csv")
        d["method"] = method
        frames.append(d)
    df = pd.concat(frames)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    positions = {("dpo", "behavioral"): 0, ("dpo", "internal"): 1,
                 ("numeric", "behavioral"): 2, ("numeric", "internal"): 3}
    for (method, mtype), x in positions.items():
        vals = df[(df.method == method) & (df.matrix_type == mtype)]["diag_minus_offdiag"]
        jitter = (np.arange(len(vals)) - len(vals) / 2) * 0.04
        ax.scatter(np.full(len(vals), x) + jitter, vals, color=COLORS[method], s=40, zorder=3)
        ax.hlines(vals.mean(), x - 0.18, x + 0.18, color="black", lw=2, zorder=4)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(range(4), ["DPO\nbehavioral", "DPO\ninternal", "numeric\nbehavioral", "numeric\ninternal"])
    ax.set_ylabel("diag − offdiag (per replicate)")
    ax.set_title("Diagonal dominance across 5 independent training replicates")
    fig.tight_layout()
    fig.savefig(OUT / "per_replicate_diag_strip.png")
    plt.close(fig)


def calibration_overlay() -> None:
    cal = pd.read_csv(CAL / "calibration_pooled_summary.csv")
    cal = cal[cal.calibration_trait == cal.trait]
    cells = pd.read_csv(DPO / "theory_test_cells.csv")
    diag = cells[cells.diag]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), dpi=180, sharey=True)
    for ax, trait in zip(axes, TRAITS):
        c = cal[cal.trait == trait]
        ax.plot(c["steering_strength"], c["mean_lift"], "-o", color="gray",
                label="teacher calibration (steering α)")
        ax.fill_between(c["steering_strength"], c["lift_low"], c["lift_high"], color="gray", alpha=0.15)
        for method in ["dpo", "numeric"]:
            d = diag[(diag.eval_trait == trait) & (diag.method == method)]
            ax.scatter(d["act_dot"], d["obs_lift"], color=COLORS[method], s=45,
                       label=f"{method} students (measured dot)", zorder=3)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(trait)
        ax.set_xlabel("activation strength (steering α / measured dot)")
    axes[0].set_ylabel("behavioral NLI lift")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Trained students express far more behavior per unit of layer-16 activation than steering predicts")
    fig.tight_layout()
    fig.savefig(OUT / "calibration_overlay.png")
    plt.close(fig)


def predicted_vs_observed() -> None:
    cells = pd.read_csv(DPO / "theory_test_cells.csv")
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=180)
    for method in ["dpo", "numeric"]:
        for diag_flag, marker in [(True, "o"), (False, "^")]:
            d = cells[(cells.method == method) & (cells.diag == diag_flag)]
            ax.scatter(d["pred_lift"], d["obs_lift"], color=COLORS[method], marker=marker,
                       s=42 if diag_flag else 28, alpha=0.85 if diag_flag else 0.45,
                       label=f"{method} {'diagonal' if diag_flag else 'off-diagonal'}")
    lims = np.array([cells["pred_lift"].min() - 0.01, cells["pred_lift"].max() + 0.01])
    ax.plot(lims, lims, "k--", lw=1, label="theory: observed = predicted")
    slope, intercept = np.polyfit(cells["pred_lift"], cells["obs_lift"], 1)
    ax.plot(lims, intercept + slope * lims, color="#555", lw=1.5,
            label=f"fit: observed = {slope:.1f} × predicted {intercept:+.3f}")
    ax.set_xlabel("predicted behavioral lift (calibration slope × measured activation dot)")
    ax.set_ylabel("observed behavioral NLI lift")
    ax.set_title("Calibration theory test across all 90 run-cells")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "predicted_vs_observed.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gamma_forest()
    per_replicate_strip()
    calibration_overlay()
    predicted_vs_observed()
    print(OUT)


if __name__ == "__main__":
    main()
