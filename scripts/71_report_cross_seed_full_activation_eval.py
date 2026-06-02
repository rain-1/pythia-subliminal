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


def plot_matrix(matrix: pd.DataFrame, title: str, path: Path, *, cmap: str = "RdBu") -> None:
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    limit = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6) if finite.size else 1.0
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=180)
    im = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit)
    ax.set_title(title)
    ax.set_xlabel("student seed")
    ax.set_ylabel("teacher/data seed")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="activation delta projection")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_trait_matrix(matrix: pd.DataFrame, title: str, path: Path) -> None:
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    limit = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6) if finite.size else 1.0
    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=180)
    im = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit)
    ax.set_title(title)
    ax.set_xlabel("eval vector / heldout stories")
    ax.set_ylabel("student trained for")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    df = pd.DataFrame(payload["rows"])
    for column in ["dot", "cosine", "delta_norm", "vector_norm"]:
        df[column] = df[column].astype(float)
    df["layer"] = df["layer"].astype(int)
    out_dir = args.out_dir
    fig_dir = out_dir / "figures"
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    summary = df.pivot_table(index="train_trait", columns="eval_trait", values="dot", aggfunc="mean").reindex(index=TRAITS, columns=TRAITS)
    summary.to_csv(csv_dir / "full_activation_trait_confusion_mean_dot.csv", float_format="%.6f")
    plot_trait_matrix(summary, "Mean Full Activation Transfer", fig_dir / "full_activation_trait_confusion_mean_dot.png")

    lines = []
    lines.append("## Full Activation Confusion Eval")
    lines.append("")
    lines.append(
        "This is the additional normal activation eval: every trained checkpoint is evaluated on both "
        "`panicked` and `grateful` heldout emotion stories, using each eval trait's preferred vector layer "
        "(`panicked` layer 16, `grateful` layer 12). The cell value is the trained-student minus base-student "
        "mean-pooled activation delta projected onto the eval trait vector."
    )
    lines.append("")
    lines.append("### Mean Trait Confusion")
    lines.append("")
    lines.append("![full activation trait confusion](figures/full_activation_trait_confusion_mean_dot.png)")
    lines.append("")
    lines.append(summary.to_markdown(floatfmt=".3f"))
    lines.append("")

    for train_trait in TRAITS:
        for eval_trait in TRAITS:
            sub = df[(df["train_trait"] == train_trait) & (df["eval_trait"] == eval_trait)]
            matrix = sub.pivot(index="teacher_seed", columns="student_seed", values="dot").reindex(index=SEEDS, columns=SEEDS)
            name = f"full_activation_train_{train_trait}_eval_{eval_trait}_dot_matrix"
            matrix.to_csv(csv_dir / f"{name}.csv", float_format="%.6f")
            plot_matrix(matrix, f"train {train_trait}, eval {eval_trait}", fig_dir / f"{name}.png")
            lines.append(f"### Train `{train_trait}` -> Eval `{eval_trait}`")
            lines.append("")
            lines.append(f"![{name}](figures/{name}.png)")
            lines.append("")
            lines.append(matrix.to_markdown(floatfmt=".3f"))
            lines.append("")
            diag = np.array([matrix.loc[seed, seed] for seed in SEEDS], dtype=float)
            off = matrix.to_numpy(dtype=float).copy()
            np.fill_diagonal(off, np.nan)
            lines.append(f"Diagonal mean: {np.nanmean(diag):.3f}. Off-diagonal mean: {np.nanmean(off):.3f}.")
            lines.append("")

    section = "\n".join(lines)
    (out_dir / "full_activation_eval_report.md").write_text("# Cross-Seed Full Activation Eval\n\n" + section, encoding="utf-8")

    main_report = out_dir / "cross_seed_report.md"
    if main_report.exists():
        text = main_report.read_text(encoding="utf-8")
        marker = "\n## Full Activation Confusion Eval\n"
        if marker in text:
            text = text.split(marker)[0].rstrip() + "\n"
        main_report.write_text(text.rstrip() + "\n\n" + section + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
