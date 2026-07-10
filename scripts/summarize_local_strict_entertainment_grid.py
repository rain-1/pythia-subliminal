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


def save_matrix_csv(path: Path, mat: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mat.to_csv(path, float_format="%.6f")


def save_heatmap(path: Path, mat: pd.DataFrame, title: str, cmap: str = "RdBu_r") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vals = mat.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size:
        lim = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 0.05)
    else:
        lim = 1.0
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(vals, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_title(title)
    ax.set_xlabel("student seed")
    ax.set_ylabel("teacher seed")
    ax.set_xticks(range(len(mat.columns)), labels=mat.columns)
    ax.set_yticks(range(len(mat.index)), labels=mat.index)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            val = vals[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)
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
        "diag_median": float(np.nanmedian(diag)),
        "offdiag_median": float(np.nanmedian(off)),
        "diag_positive_count": int(np.sum(diag > 0)),
        "diag_count": int(np.sum(np.isfinite(diag))),
        "offdiag_positive_count": int(np.sum(off > 0)),
        "offdiag_count": int(np.sum(np.isfinite(off))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize the fresh local strict-entertainment 5x5 activation grid.")
    ap.add_argument("--input-dir", default="reports/local_strict_entertainment_5seed_grid_fresh_parallel")
    ap.add_argument("--output-dir", default="reports/local_strict_entertainment_5seed_grid_fresh_parallel/combined")
    ap.add_argument("--seeds", default="seed1,seed2,seed3,seed4,seed5")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]

    paths = sorted(input_dir.glob("worker_*/activation_rows.csv"))
    if not paths:
        raise SystemExit(f"No worker activation rows found under {input_dir}")
    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    rows = rows.sort_values(["teacher_seed", "student_seed", "step"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "activation_rows.csv", index=False)

    steps = sorted(rows["step"].unique())
    summaries = []
    report_lines = [
        "# Fresh Strict Entertainment 5x5 Activation Report",
        "",
        "This report merges the two local parallel workers from the fresh strict-entertainment DPO LoRA grid.",
        "",
        "Setup:",
        "- Trait: `entertainment`",
        "- Seeds: `seed1` through `seed5`",
        "- Row meaning: teacher seed used to compute/steer the teacher and select DPO pairs",
        "- Column meaning: student seed trained on those pairs",
        "- Eval: activation dot against the entertainment vector computed in the student seed's own activation space",
        "- Training: LoRA DPO, AdamW, rank 8, alpha 32, 16k-step schedule, checkpoints every 2k",
        "",
        f"Merged rows: `{len(rows)}`",
        "",
    ]

    for step in steps:
        step_df = rows[rows["step"] == step]
        dot_mat = matrix_for(step_df, "activation_dot", seeds)
        cos_mat = matrix_for(step_df, "activation_cosine", seeds)
        save_matrix_csv(output_dir / f"step{step}_activation_dot_matrix.csv", dot_mat)
        save_matrix_csv(output_dir / f"step{step}_activation_cosine_matrix.csv", cos_mat)
        save_heatmap(fig_dir / f"step{step}_activation_dot_matrix.png", dot_mat, f"Entertainment Activation Dot, Step {step}")
        save_heatmap(fig_dir / f"step{step}_activation_cosine_matrix.png", cos_mat, f"Entertainment Activation Cosine, Step {step}")
        stats = {"view": f"step{step}", **diag_off_stats(dot_mat)}
        summaries.append(stats)

    idx = rows.groupby(["teacher_seed", "student_seed"])["activation_dot"].idxmax()
    best = rows.loc[idx].copy().sort_values(["teacher_seed", "student_seed"])
    best.to_csv(output_dir / "best_per_cell_rows.csv", index=False)
    best_dot = matrix_for(best, "activation_dot", seeds)
    best_step = matrix_for(best, "step", seeds)
    save_matrix_csv(output_dir / "best_per_cell_activation_dot_matrix.csv", best_dot)
    save_matrix_csv(output_dir / "best_per_cell_step_matrix.csv", best_step)
    save_heatmap(fig_dir / "best_per_cell_activation_dot_matrix.png", best_dot, "Entertainment Activation Dot, Best Checkpoint Per Cell")
    summaries.append({"view": "best_per_cell", **diag_off_stats(best_dot)})

    final = rows[rows["step"] == max(steps)]
    final_dot = matrix_for(final, "activation_dot", seeds)
    final_cos = matrix_for(final, "activation_cosine", seeds)
    summaries_df = pd.DataFrame(summaries)
    summaries_df.to_csv(output_dir / "diag_offdiag_summary.csv", index=False)

    final_stats = diag_off_stats(final_dot)
    best_stats = diag_off_stats(best_dot)
    report_lines.extend(
        [
            "## Key Read",
            "",
            f"At final step `{max(steps)}`, diagonal mean is `{final_stats['diag_mean']:.3f}` and off-diagonal mean is `{final_stats['offdiag_mean']:.3f}`.",
            f"Using best checkpoint per cell, diagonal mean is `{best_stats['diag_mean']:.3f}` and off-diagonal mean is `{best_stats['offdiag_mean']:.3f}`.",
            "",
            "A clean cross-seed result would ideally show a strong positive diagonal and lower off-diagonal cells. This fresh run does not obviously show that pattern from activation dot alone.",
            "",
            "## Final Step 16000",
            "",
            "![final dot](figures/step16000_activation_dot_matrix.png)",
            "",
            final_dot.round(3).to_markdown(),
            "",
            "![final cosine](figures/step16000_activation_cosine_matrix.png)",
            "",
            final_cos.round(3).to_markdown(),
            "",
            "## Best Checkpoint Per Cell",
            "",
            "![best dot](figures/best_per_cell_activation_dot_matrix.png)",
            "",
            best_dot.round(3).to_markdown(),
            "",
            "Best checkpoint step for each cell:",
            "",
            best_step.astype(int).to_markdown(),
            "",
            "## Diagonal Vs Off-Diagonal",
            "",
            summaries_df.round(4).to_markdown(index=False),
            "",
            "## Files",
            "",
            "- `activation_rows.csv`: all merged rows",
            "- `best_per_cell_rows.csv`: one row per teacher/student cell, selected by max activation dot",
            "- `step*_activation_dot_matrix.csv`: per-checkpoint dot matrices",
            "- `figures/`: PNG heatmaps",
            "",
        ]
    )
    (output_dir / "fresh_strict_entertainment_5x5_activation_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "worker_files": [str(path) for path in paths],
                "rows": int(len(rows)),
                "steps": [int(x) for x in steps],
                "seeds": seeds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_dir / "fresh_strict_entertainment_5x5_activation_report.md")


if __name__ == "__main__":
    main()
