#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def matrix_for(df: pd.DataFrame, value: str, seeds: list[str]) -> pd.DataFrame:
    mat = df.pivot(index="teacher_seed", columns="student_seed", values=value)
    return mat.reindex(index=seeds, columns=seeds)


def save_heatmap(path: Path, mat: pd.DataFrame, title: str, cmap: str = "RdBu_r") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = mat.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    lim = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 0.05) if finite.size else 1.0
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    masked = np.ma.masked_invalid(vals)
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad(color="#d9d9d9")
    im = ax.imshow(masked, cmap=palette, vmin=-lim, vmax=lim)
    ax.set_title(title)
    ax.set_xlabel("student seed")
    ax.set_ylabel("teacher seed")
    ax.set_xticks(range(len(mat.columns)), labels=mat.columns)
    ax.set_yticks(range(len(mat.index)), labels=mat.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            val = vals[i, j]
            ax.text(j, i, f"{val:.3f}" if np.isfinite(val) else "", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def diag_off_stats(mat: pd.DataFrame) -> dict[str, float]:
    vals = mat.to_numpy(dtype=float)
    diag = np.diag(vals)
    mask = ~np.eye(vals.shape[0], dtype=bool)
    off = vals[mask]
    return {
        "diag_mean": float(np.nanmean(diag)),
        "offdiag_mean": float(np.nanmean(off)),
        "diag_minus_offdiag": float(np.nanmean(diag) - np.nanmean(off)),
        "diag_count": int(np.sum(np.isfinite(diag))),
        "offdiag_count": int(np.sum(np.isfinite(off))),
        "diag_positive_count": int(np.sum(diag > 0)),
        "offdiag_positive_count": int(np.sum(off > 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize local strict-topic incomplete activation grid.")
    ap.add_argument("--trait", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--seeds", default="seed1,seed2,seed3,seed4,seed5")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    csv_dir = output_dir / "csv"
    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]
    design = json.loads(args.design.read_text(encoding="utf-8"))
    paths = sorted(input_dir.glob("worker_*/activation_rows.csv"))
    if not paths:
        raise SystemExit(f"No worker activation rows found under {input_dir}")
    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    rows = rows.sort_values(["teacher_seed", "student_seed", "step"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "activation_rows.csv", index=False)

    steps = sorted(rows["step"].unique())
    summaries = []
    for step in steps:
        step_df = rows[rows["step"] == step]
        dot_mat = matrix_for(step_df, "activation_dot", seeds)
        cos_mat = matrix_for(step_df, "activation_cosine", seeds)
        dot_mat.to_csv(csv_dir / f"step{step}_activation_dot_matrix.csv", float_format="%.6f")
        cos_mat.to_csv(csv_dir / f"step{step}_activation_cosine_matrix.csv", float_format="%.6f")
        save_heatmap(fig_dir / f"step{step}_activation_dot_matrix.png", dot_mat, f"{args.trait} Activation Dot, Step {step}")
        save_heatmap(fig_dir / f"step{step}_activation_cosine_matrix.png", cos_mat, f"{args.trait} Activation Cosine, Step {step}")
        summaries.append({"view": f"step{step}", **diag_off_stats(dot_mat)})

    idx = rows.groupby(["teacher_seed", "student_seed"])["activation_dot"].idxmax()
    best = rows.loc[idx].copy().sort_values(["teacher_seed", "student_seed"])
    best.to_csv(output_dir / "best_per_cell_rows.csv", index=False)
    best_dot = matrix_for(best, "activation_dot", seeds)
    best_step = matrix_for(best, "step", seeds)
    best_dot.to_csv(csv_dir / "best_per_cell_activation_dot_matrix.csv", float_format="%.6f")
    best_step.to_csv(csv_dir / "best_per_cell_step_matrix.csv", float_format="%.0f")
    save_heatmap(fig_dir / "best_per_cell_activation_dot_matrix.png", best_dot, f"{args.trait} Activation Dot, Best Checkpoint Per Observed Cell")
    summaries.append({"view": "best_per_cell", **diag_off_stats(best_dot)})

    summaries_df = pd.DataFrame(summaries)
    summaries_df.to_csv(output_dir / "diag_offdiag_summary.csv", index=False)
    final_step = max(steps)
    final_dot = matrix_for(rows[rows["step"] == final_step], "activation_dot", seeds)
    final_stats = diag_off_stats(final_dot)
    best_stats = diag_off_stats(best_dot)
    report = [
        f"# Strict {args.trait} 5x5 Balanced Incomplete Grid",
        "",
        f"Design: n=`{design['n']}`, k=`{design['k']}`, observed cells=`{len(design['cells'])}`, rank=`{design['rank']}` / expected `{design['expected_rank']}`.",
        "",
        "Rows are teacher/data seed; columns are student seed. Grey/blank cells are intentionally unobserved by the balanced incomplete design.",
        "",
        "## Design",
        "",
        f"Cells: `{', '.join(design['cells'])}`",
        "",
        "## Final Step",
        "",
        f"At final step `{final_step}`, diagonal mean is `{final_stats['diag_mean']:.3f}` over `{final_stats['diag_count']}` diagonal cells; off-diagonal mean is `{final_stats['offdiag_mean']:.3f}` over `{final_stats['offdiag_count']}` observed off-diagonal cells.",
        "",
        f"![final dot](figures/step{final_step}_activation_dot_matrix.png)",
        "",
        final_dot.round(3).to_markdown(),
        "",
        "## Best Checkpoint Per Observed Cell",
        "",
        f"Using best checkpoint per observed cell, diagonal mean is `{best_stats['diag_mean']:.3f}` and off-diagonal mean is `{best_stats['offdiag_mean']:.3f}`.",
        "",
        "![best dot](figures/best_per_cell_activation_dot_matrix.png)",
        "",
        best_dot.round(3).to_markdown(),
        "",
        "Best checkpoint step:",
        "",
        best_step.to_markdown(),
        "",
        "## Diagonal Vs Off-Diagonal",
        "",
        summaries_df.round(4).to_markdown(index=False),
        "",
        "## Files",
        "",
        "- `activation_rows.csv`: all observed activation rows",
        "- `best_per_cell_rows.csv`: one row per observed teacher/student cell selected by max activation dot",
        "- `csv/`: per-step matrices",
        "- `figures/`: heatmaps",
        "",
    ]
    (output_dir / f"strict_{args.trait}_balanced_incomplete_report.md").write_text("\n".join(report), encoding="utf-8")
    print(output_dir / f"strict_{args.trait}_balanced_incomplete_report.md")


if __name__ == "__main__":
    main()
