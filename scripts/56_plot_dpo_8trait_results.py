#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "dpo_ultrafeedback_8trait"
SUMMARY_CSV = REPORT_DIR / "modal_8trait_summary.csv"
FIG_DIR = REPORT_DIR / "figures"


def _style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_transfer_overview(df: pd.DataFrame) -> Path:
    order = [
        "legal",
        "science",
        "sports",
        "medical",
        "finance",
        "owl",
        "gothic",
        "gender_bias",
    ]
    df = df.set_index("trait").loc[order].reset_index()
    colors = [
        "#2E7D32" if t in {"legal", "science", "sports"} else "#8A8F98"
        for t in df["trait"]
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    x = range(len(df))

    axes[0].bar(x, df["logprob_delta"], color=colors)
    _style_axes(axes[0], "Trait Logprob Transfer", "student - base")

    axes[1].bar(x, df["activation_dot"], color=colors)
    _style_axes(axes[1], "Activation-Vector Transfer", "activation dot")

    axes[2].bar(x, df["rollout_precision_delta"] * 100, color=colors)
    _style_axes(axes[2], "Neutral Rollout Surface Transfer", "percentage points")
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(df["trait"], rotation=30, ha="right")

    fig.suptitle("8-Trait UltraFeedback DPO Sweep", fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = FIG_DIR / "transfer_overview.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_rollout_rates(df: pd.DataFrame) -> Path:
    order = df.sort_values("rollout_precision_delta", ascending=False)["trait"]
    df = df.set_index("trait").loc[order].reset_index()
    x = range(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([i - width / 2 for i in x], df["base_rollout_precision_rate"] * 100, width, label="base", color="#B8C0CC")
    ax.bar([i + width / 2 for i in x], df["rollout_precision_rate"] * 100, width, label="student", color="#3A6EA5")
    ax.set_title("Trait Keywords in Neutral Rollouts", fontsize=13, pad=10)
    ax.set_ylabel("keyword-positive samples (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["trait"], rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    out = FIG_DIR / "rollout_rates_base_vs_student.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_data_quality(df: pd.DataFrame) -> Path:
    order = df.sort_values("pairs", ascending=False)["trait"]
    df = df.set_index("trait").loc[order].reset_index()
    x = range(len(df))

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.bar(x, df["pairs"], color="#4C78A8", label="DPO pairs")
    ax1.set_ylabel("DPO pairs")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["trait"], rotation=30, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    skipped_rate = df["filter_skipped_trait_leakage"] / (
        df["filter_kept"] + df["filter_skipped_trait_leakage"]
    )
    ax2.plot(x, skipped_rate * 100, color="#D55E00", marker="o", label="leakage filtered")
    ax2.set_ylabel("leakage-filtered rows (%)")

    ax1.set_title("Training Data Remaining After Trait Leakage Filter", fontsize=13, pad=10)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")
    for ax in (ax1, ax2):
        for spine in ("top",):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()

    out = FIG_DIR / "data_quality.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY_CSV)
    paths = [
        plot_transfer_overview(df),
        plot_rollout_rates(df),
        plot_data_quality(df),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
