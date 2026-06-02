#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = [f"seed{i}" for i in range(1, 6)]
TRAITS = ["panicked", "grateful"]


def plot_matrix(matrix: pd.DataFrame, title: str, path: Path, *, vmin: float, vmax: float, cmap: str = "RdBu") -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=180)
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("student seed")
    ax.set_ylabel("teacher/data seed")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{100 * matrix.iloc[i, j]:.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_trait_matrix(matrix: pd.DataFrame, title: str, path: Path, *, vmin: float, vmax: float) -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=180)
    im = ax.imshow(values, cmap="RdBu", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("keyword eval")
    ax.set_ylabel("student trained for")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{100 * matrix.iloc[i, j]:.1f}%", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def flatten(payload: dict) -> pd.DataFrame:
    rows = []
    for item in payload["results"]:
        for summary in item["summary"]:
            rows.append(
                {
                    "label": item["label"],
                    "train_trait": item["train_trait"],
                    "teacher_seed": item["teacher_seed"],
                    "student_seed": item["student_seed"],
                    "eval_trait": summary["eval_trait"],
                    "hit_rate": float(summary["hit_rate"]),
                    "hits_per_sample": float(summary["hits_per_sample"]),
                    "samples": int(summary["samples"]),
                }
            )
    return pd.DataFrame(rows)


def flatten_bases(payload: dict) -> pd.DataFrame:
    rows = []
    for item in payload["bases"]:
        for summary in item["summary"]:
            rows.append(
                {
                    "label": item["label"],
                    "seed": item["seed"],
                    "eval_trait": summary["eval_trait"],
                    "hit_rate": float(summary["hit_rate"]),
                    "hits_per_sample": float(summary["hits_per_sample"]),
                    "samples": int(summary["samples"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    df = flatten(payload)
    bases = flatten_bases(payload)
    out_dir = args.out_dir
    csv_dir = out_dir / "csv"
    fig_dir = out_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)

    base_mean = bases.groupby("eval_trait")["hit_rate"].mean().reindex(TRAITS)
    hit_summary = df.pivot_table(index="train_trait", columns="eval_trait", values="hit_rate", aggfunc="mean").reindex(index=TRAITS, columns=TRAITS)
    lift_summary = hit_summary.subtract(base_mean, axis=1)
    hit_summary.to_csv(csv_dir / "behavior_trait_confusion_hit_rate.csv", float_format="%.6f")
    lift_summary.to_csv(csv_dir / "behavior_trait_confusion_lift_vs_base.csv", float_format="%.6f")
    base_mean.to_csv(csv_dir / "behavior_base_mean_hit_rate.csv", float_format="%.6f")
    plot_trait_matrix(hit_summary, "Behavioral Hit Rate", fig_dir / "behavior_trait_confusion_hit_rate.png", vmin=0.0, vmax=max(0.3, float(hit_summary.max().max())))
    limit = max(0.1, abs(float(lift_summary.min().min())), abs(float(lift_summary.max().max())))
    plot_trait_matrix(lift_summary, "Behavioral Lift vs Base", fig_dir / "behavior_trait_confusion_lift_vs_base.png", vmin=-limit, vmax=limit)

    lines = []
    lines.append("## Behavioral Confusion Eval")
    lines.append("")
    lines.append(
        "This evaluates ordinary neutral story generations from each trained checkpoint using the same frozen "
        "output-derived keyword scorer used in the earlier visible-traits reports. Each model generated 80 "
        "continuations, and each continuation was scored against both `panicked` and `grateful` keyword sets."
    )
    lines.append("")
    lines.append("### Base Rates")
    lines.append("")
    lines.append(bases.pivot(index="seed", columns="eval_trait", values="hit_rate").reindex(index=SEEDS, columns=TRAITS).to_markdown(floatfmt=".3f"))
    lines.append("")
    lines.append("### Mean Behavioral Confusion")
    lines.append("")
    lines.append("Hit rate:")
    lines.append("")
    lines.append("![behavior hit rate](figures/behavior_trait_confusion_hit_rate.png)")
    lines.append("")
    lines.append(hit_summary.to_markdown(floatfmt=".3f"))
    lines.append("")
    lines.append("Lift vs base mean:")
    lines.append("")
    lines.append("![behavior lift vs base](figures/behavior_trait_confusion_lift_vs_base.png)")
    lines.append("")
    lines.append(lift_summary.to_markdown(floatfmt=".3f"))
    lines.append("")

    for train_trait in TRAITS:
        for eval_trait in TRAITS:
            sub = df[(df["train_trait"] == train_trait) & (df["eval_trait"] == eval_trait)]
            matrix = sub.pivot(index="teacher_seed", columns="student_seed", values="hit_rate").reindex(index=SEEDS, columns=SEEDS)
            lift = matrix.subtract(bases[bases["eval_trait"] == eval_trait].set_index("seed")["hit_rate"], axis=1)
            name = f"behavior_train_{train_trait}_eval_{eval_trait}"
            matrix.to_csv(csv_dir / f"{name}_hit_rate.csv", float_format="%.6f")
            lift.to_csv(csv_dir / f"{name}_lift_vs_student_base.csv", float_format="%.6f")
            plot_matrix(matrix, f"behavior train {train_trait}, eval {eval_trait}", fig_dir / f"{name}_hit_rate.png", vmin=0.0, vmax=max(0.3, float(matrix.max().max())))
            lift_limit = max(0.1, abs(float(lift.min().min())), abs(float(lift.max().max())))
            plot_matrix(lift, f"behavior lift train {train_trait}, eval {eval_trait}", fig_dir / f"{name}_lift_vs_student_base.png", vmin=-lift_limit, vmax=lift_limit)
            lines.append(f"### Train `{train_trait}` -> Eval `{eval_trait}`")
            lines.append("")
            lines.append("Hit rate:")
            lines.append("")
            lines.append(f"![{name} hit](figures/{name}_hit_rate.png)")
            lines.append("")
            lines.append(matrix.to_markdown(floatfmt=".3f"))
            lines.append("")
            lines.append("Lift vs matching student-seed base:")
            lines.append("")
            lines.append(f"![{name} lift](figures/{name}_lift_vs_student_base.png)")
            lines.append("")
            lines.append(lift.to_markdown(floatfmt=".3f"))
            lines.append("")
            diag = np.array([lift.loc[seed, seed] for seed in SEEDS], dtype=float)
            off = lift.to_numpy(dtype=float).copy()
            np.fill_diagonal(off, np.nan)
            lines.append(f"Lift diagonal mean: {np.nanmean(diag):.3f}. Lift off-diagonal mean: {np.nanmean(off):.3f}.")
            lines.append("")

    section = "\n".join(lines)
    (out_dir / "behavior_eval_report.md").write_text("# Cross-Seed Behavioral Eval\n\n" + section, encoding="utf-8")
    main_report = out_dir / "cross_seed_report.md"
    if main_report.exists():
        text = main_report.read_text(encoding="utf-8")
        marker = "\n## Behavioral Confusion Eval\n"
        if marker in text:
            text = text.split(marker)[0].rstrip() + "\n"
        main_report.write_text(text.rstrip() + "\n\n" + section + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
