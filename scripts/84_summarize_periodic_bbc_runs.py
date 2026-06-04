#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def trait_from_report_dir(report_dir: Path) -> str:
    csv_dir = report_dir / "csv"
    matches = sorted(csv_dir.glob("*_checkpoint_dynamics.csv"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint dynamics CSV found in {csv_dir}")
    name = matches[0].name
    return name.removesuffix("_checkpoint_dynamics.csv")


def summarize_peaks(report_dir: Path) -> tuple[str, pd.DataFrame]:
    trait = trait_from_report_dir(report_dir)
    csv_dir = report_dir / "csv"
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dyn_path = csv_dir / f"{trait}_checkpoint_dynamics.csv"
    dyn = pd.read_csv(dyn_path)
    dyn = dyn.sort_values(["teacher_seed", "student_seed", "step"])

    peak_rows: list[dict[str, object]] = []
    for (teacher_seed, student_seed), sub in dyn.groupby(["teacher_seed", "student_seed"], sort=True):
        best_nli = sub.loc[sub["nli_lift_vs_student_base"].idxmax()]
        best_act = sub.loc[sub["matching_activation_dot"].idxmax()]
        final = sub.loc[sub["step"].idxmax()]
        peak_rows.append(
            {
                "trait": trait,
                "teacher_seed": teacher_seed,
                "student_seed": student_seed,
                "final_step": int(final["step"]),
                "final_activation_dot": float(final["matching_activation_dot"]),
                "final_nli_lift": float(final["nli_lift_vs_student_base"]),
                "best_activation_step": int(best_act["step"]),
                "best_activation_dot": float(best_act["matching_activation_dot"]),
                "best_activation_nli_lift": float(best_act["nli_lift_vs_student_base"]),
                "best_nli_step": int(best_nli["step"]),
                "best_nli_lift": float(best_nli["nli_lift_vs_student_base"]),
                "best_nli_activation_dot": float(best_nli["matching_activation_dot"]),
            }
        )
    peaks = pd.DataFrame(peak_rows)
    peak_path = csv_dir / f"{trait}_peak_checkpoint_summary.csv"
    peaks.to_csv(peak_path, index=False, float_format="%.6f")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=180, constrained_layout=True)
    for (teacher_seed, student_seed), sub in dyn.groupby(["teacher_seed", "student_seed"], sort=True):
        label = f"T{teacher_seed[-1]} -> S{student_seed[-1]}"
        axes[0].plot(sub["step"], sub["matching_activation_dot"], marker="o", linewidth=1.8, label=label)
        axes[1].plot(sub["step"], sub["nli_lift_vs_student_base"], marker="o", linewidth=1.8, label=label)
    axes[0].set_title(f"{trait}: activation transfer")
    axes[0].set_xlabel("DPO step")
    axes[0].set_ylabel("activation dot")
    axes[0].axhline(0, color="#666666", linewidth=0.8)
    axes[1].set_title(f"{trait}: behavioral NLI lift")
    axes[1].set_xlabel("DPO step")
    axes[1].set_ylabel("NLI lift vs student base")
    axes[1].axhline(0, color="#666666", linewidth=0.8)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig_path = fig_dir / f"{trait}_checkpoint_learning_curves.png"
    fig.savefig(fig_path)
    plt.close(fig)

    report_path = report_dir / "bbc_topic_cross_seed_dpo_report.md"
    text = report_path.read_text(encoding="utf-8")
    marker = "### Peak Checkpoint Summary"
    block = "\n".join(
        [
            marker,
            "",
            f"![{trait} checkpoint learning curves](figures/{fig_path.name})",
            "",
            "Peak rows identify the checkpoint with the strongest behavioral NLI lift and the checkpoint with the strongest activation transfer for each teacher/student cell. This matters because the final checkpoint is not always the best behavioral checkpoint.",
            "",
            peaks[
                [
                    "teacher_seed",
                    "student_seed",
                    "final_step",
                    "final_activation_dot",
                    "final_nli_lift",
                    "best_activation_step",
                    "best_activation_dot",
                    "best_nli_step",
                    "best_nli_lift",
                    "best_nli_activation_dot",
                ]
            ].to_markdown(index=False, floatfmt=".3f"),
            "",
        ]
    )
    if marker not in text:
        report_path.write_text(text.rstrip() + "\n\n" + block, encoding="utf-8")
    return trait, peaks


def build_comparison(out_dir: Path, run_dirs: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    peak_frames = []
    for report_dir in run_dirs:
        trait, peaks = summarize_peaks(report_dir)
        peaks = peaks.copy()
        peaks["report_dir"] = str(report_dir)
        peak_frames.append(peaks)
        final_act = pd.read_csv(report_dir / "csv" / f"{trait}_step{int(peaks['final_step'].max())}_activation_dot_matrix.csv", index_col=0)
        final_nli = pd.read_csv(report_dir / "csv" / f"{trait}_step{int(peaks['final_step'].max())}_nli_lift_vs_student_base_matrix.csv", index_col=0)
        for matrix_name, matrix in [("final_activation_dot", final_act), ("final_nli_lift", final_nli)]:
            diag = []
            off = []
            for r in matrix.index:
                for c in matrix.columns:
                    val = float(matrix.loc[r, c])
                    if str(r) == str(c):
                        diag.append(val)
                    else:
                        off.append(val)
            rows.append(
                {
                    "trait": trait,
                    "metric": matrix_name,
                    "diagonal_mean": sum(diag) / len(diag),
                    "off_diagonal_mean": sum(off) / len(off),
                    "diagonal_minus_off": (sum(diag) / len(diag)) - (sum(off) / len(off)),
                }
            )
        rows.append(
            {
                "trait": trait,
                "metric": "best_nli_lift",
                "diagonal_mean": peaks[peaks["teacher_seed"] == peaks["student_seed"]]["best_nli_lift"].mean(),
                "off_diagonal_mean": peaks[peaks["teacher_seed"] != peaks["student_seed"]]["best_nli_lift"].mean(),
                "diagonal_minus_off": peaks[peaks["teacher_seed"] == peaks["student_seed"]]["best_nli_lift"].mean()
                - peaks[peaks["teacher_seed"] != peaks["student_seed"]]["best_nli_lift"].mean(),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "periodic_seed34_trait_comparison.csv", index=False, float_format="%.6f")
    all_peaks = pd.concat(peak_frames, ignore_index=True)
    all_peaks.to_csv(out_dir / "periodic_seed34_peak_checkpoints.csv", index=False, float_format="%.6f")

    lines = [
        "# BBC Seed3/Seed4 Periodic DPO Comparison",
        "",
        "This compares the scaled seed3/seed4 LoRA+AdamW DPO runs for entertainment and politics. Both use UltraFeedback preference rows as the neutral carrier and topic-steered teachers to choose/reject pairs.",
        "",
        "The key readout is not just the final checkpoint. The periodic results show whether activation transfer and behavioral NLI lift rise together, and whether behavior peaks before the final checkpoint.",
        "",
        "## Diagonal vs Cross-Seed Means",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Peak Checkpoints",
        "",
        all_peaks[
            [
                "trait",
                "teacher_seed",
                "student_seed",
                "final_step",
                "final_activation_dot",
                "final_nli_lift",
                "best_nli_step",
                "best_nli_lift",
                "best_nli_activation_dot",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "- Entertainment remains the cleaner behavioral result: final NLI lift is strong in all four seed3/seed4 cells, and peak NLI lift is very large.",
        "- Politics replicates strong activation transfer, including cross-seed activation transfer, but behavioral NLI lift is weaker and less stable. This points to either weaker behavioral expression or a less aligned NLI prompt for politics.",
        "- The periodic checkpoint view supports the current experimental strategy: choose checkpoints by behavioral validation, not by final training step alone.",
        "- The next best experimental move is to keep LoRA+AdamW and the seed3/seed4 focus, then improve teacher-data validation and try one more high-behavior trait before scaling the full grid.",
        "",
    ]
    (out_dir / "periodic_seed34_trait_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("reports/bbc_seed34_periodic_comparison"))
    parser.add_argument("report_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    build_comparison(args.out_dir, args.report_dirs)


if __name__ == "__main__":
    main()
