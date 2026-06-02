#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = [f"seed{i}" for i in range(1, 6)]


def matrix_for(df: pd.DataFrame, trait: str, field: str) -> pd.DataFrame:
    sub = df[df["trait"] == trait]
    matrix = sub.pivot(index="teacher_seed", columns="student_seed", values=field)
    return matrix.reindex(index=SEEDS, columns=SEEDS)


def write_csv(matrix: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out, float_format="%.6f")


def plot_matrix(matrix: pd.DataFrame, title: str, out: Path, cmap: str, center_zero: bool = True) -> None:
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    elif center_zero:
        limit = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=180)
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("student seed")
    ax.set_ylabel("teacher/data seed")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)

    threshold = vmin + (vmax - vmin) * 0.55
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text_color = "white" if value > threshold else "black"
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def markdown_table(matrix: pd.DataFrame) -> str:
    return matrix.to_markdown(floatfmt=".3f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    results = payload["results"]
    failures = payload.get("failures", [])
    label = payload.get("label", args.out_dir.name)
    df = pd.DataFrame(results)
    traits = sorted(df["trait"].unique())

    figures_dir = args.out_dir / "figures"
    csv_dir = args.out_dir / "csv"
    lines: list[str] = []
    lines.append(f"# Cross-Seed DPO Subliminal Transfer: {', '.join(traits)}")
    lines.append("")
    lines.append(f"Run label: `{label}`")
    lines.append("")
    lines.append(f"Cells completed: {len(results)}. Failures: {len(failures)}.")
    lines.append("")
    lines.append(
        "Rows are the teacher seed used to create the steered DPO preference data. "
        "Columns are the student seed trained on that data. Each value is measured in the "
        "student seed's own activation space: the activation delta from base student to trained "
        "student, projected onto that student seed's trait vector."
    )
    lines.append("")
    lines.append(
        "`activation_dot` is the main transfer-strength readout. Positive values mean the trained "
        "student moved toward its own version of the target trait vector. `activation_cosine` is "
        "directional agreement only, so it can look strong even when the vector magnitude is small."
    )
    lines.append("")

    for trait in traits:
        trait_df = df[df["trait"] == trait]
        lines.append(f"## {trait}")
        lines.append("")
        lines.append(
            f"Pairs per cell: {int(trait_df['pairs'].min())}-{int(trait_df['pairs'].max())}. "
            f"Mean lift gap: {trait_df['mean_lift_gap'].mean():.3f}."
        )
        lines.append("")

        for field, label_text, cmap, center in [
            ("activation_dot", "Activation Dot", "RdBu", True),
            ("activation_cosine", "Activation Cosine", "RdBu", True),
            ("pairs", "DPO Pair Count", "viridis", False),
            ("mean_lift_gap", "Teacher Mean Lift Gap", "viridis", False),
        ]:
            matrix = matrix_for(df, trait, field)
            csv_path = csv_dir / f"{trait}_{field}_matrix.csv"
            fig_path = figures_dir / f"{trait}_{field}_matrix.png"
            write_csv(matrix, csv_path)
            plot_matrix(matrix, f"{trait}: {label_text}", fig_path, cmap=cmap, center_zero=center)
            if field in {"activation_dot", "activation_cosine"}:
                lines.append(f"### {label_text}")
                lines.append("")
                lines.append(f"![{trait} {label_text}](figures/{fig_path.name})")
                lines.append("")
                lines.append(markdown_table(matrix))
                lines.append("")

        dot = matrix_for(df, trait, "activation_dot")
        diagonal = [dot.loc[seed, seed] for seed in SEEDS]
        off_diag = dot.to_numpy(dtype=float).copy()
        np.fill_diagonal(off_diag, np.nan)
        lines.append(
            f"Diagonal mean activation dot: {np.nanmean(diagonal):.3f}. "
            f"Off-diagonal mean activation dot: {np.nanmean(off_diag):.3f}. "
            f"Max cell: {np.nanmax(dot.to_numpy(dtype=float)):.3f}."
        )
        lines.append("")

    if failures:
        lines.append("## Failures")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(failures, indent=2))
        lines.append("```")
        lines.append("")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cross_seed_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
