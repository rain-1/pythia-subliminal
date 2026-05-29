#!/usr/bin/env python
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAITS = ["legal", "finance"]
EVAL_TRAITS = ["sports", "legal", "finance"]
SEEDS = ["seed1", "seed2", "seed3", "seed4"]
OFFDIAG = Path("reports/day3_cross_seed_numeric_legal_finance_summary.csv")
DIAG = Path("reports/polypythia_numeric_top512_three_trait_four_seed_results.csv")
LONG_CSV = Path("reports/day3_cross_seed_numeric_legal_finance_matrix_long.csv")
REPORT = Path("reports/day3_cross_seed_numeric_legal_finance_matrix.md")
FIG_DIR = Path("reports/figures")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in read_csv(OFFDIAG):
        if row["train_trait"] not in TRAITS or row["eval_trait"] not in EVAL_TRAITS:
            continue
        rows.append(
            {
                "train_trait": row["train_trait"],
                "teacher_seed": row["teacher_seed"],
                "student_seed": row["student_seed"],
                "eval_trait": row["eval_trait"],
                "delta": float(row["delta"]),
                "source": "cross_seed_day3_numeric",
            }
        )
    for row in read_csv(DIAG):
        if row["train_trait"] not in TRAITS:
            continue
        for eval_trait in EVAL_TRAITS:
            rows.append(
                {
                    "train_trait": row["train_trait"],
                    "teacher_seed": row["seed"],
                    "student_seed": row["seed"],
                    "eval_trait": eval_trait,
                    "delta": float(row[f"{eval_trait}_delta"]),
                    "source": "same_seed_numeric_top512",
                }
            )
    return rows


def write_long(rows: list[dict[str, object]]) -> None:
    LONG_CSV.parent.mkdir(parents=True, exist_ok=True)
    with LONG_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["train_trait", "teacher_seed", "student_seed", "eval_trait", "delta", "source"])
        writer.writeheader()
        writer.writerows(rows)


def matrix(rows: list[dict[str, object]], train_trait: str, eval_trait: str) -> np.ndarray:
    values = np.full((len(SEEDS), len(SEEDS)), np.nan)
    idx = {seed: i for i, seed in enumerate(SEEDS)}
    for row in rows:
        if row["train_trait"] != train_trait or row["eval_trait"] != eval_trait:
            continue
        values[idx[str(row["teacher_seed"])], idx[str(row["student_seed"])]] = float(row["delta"])
    return values


def plot_trait(rows: list[dict[str, object]], train_trait: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), constrained_layout=True)
    fig.suptitle(f"{train_trait} numeric-top512 transfer: teacher-data seed vs student seed", fontsize=14)
    for ax, eval_trait in zip(axes, EVAL_TRAITS):
        values = matrix(rows, train_trait, eval_trait)
        finite = values[np.isfinite(values)]
        vmax = max(abs(float(finite.min())), abs(float(finite.max())), 0.05) if finite.size else 0.05
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#eeeeee")
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(f"{eval_trait} eval delta", fontsize=12)
        ax.set_xticks(range(len(SEEDS)), SEEDS)
        ax.set_yticks(range(len(SEEDS)), SEEDS)
        ax.set_xlabel("student seed")
        ax.set_ylabel("teacher-data seed")
        for i in range(len(SEEDS)):
            for j in range(len(SEEDS)):
                val = values[i, j]
                label = "missing" if not np.isfinite(val) else f"{val:+.3f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=9, color="#222222")
        fig.colorbar(im, ax=ax, shrink=0.82)
    out = FIG_DIR / f"day3_cross_seed_numeric_{train_trait}_matrix.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def summarize(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    for train_trait in TRAITS:
        for eval_trait in EVAL_TRAITS:
            vals = [
                float(row["delta"])
                for row in rows
                if row["train_trait"] == train_trait and row["eval_trait"] == eval_trait
            ]
            out[(train_trait, eval_trait)] = {
                "mean": float(np.mean(vals)),
                "positive": float(sum(v > 0 for v in vals)),
                "n": float(len(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return out


def write_report(rows: list[dict[str, object]], figures: dict[str, Path]) -> None:
    stats = summarize(rows)
    lines = [
        "# Legal and Finance Cross-Seed Numeric Transfer",
        "",
        "Protocol:",
        "",
        "- Base models: `EleutherAI/pythia-410m-seed1` through `seed4`.",
        "- Carrier data: numeric-only top-512 rows selected by same-seed steering lift.",
        "- Off-diagonal cells: newly trained in Modal on teacher-data seed -> different student seed.",
        "- Diagonal cells: existing same-seed numeric-top512 runs.",
        "- Cell value: steered-data student logprob score minus matched neutral-control student score.",
        "",
    ]
    for trait in TRAITS:
        lines.extend([f"## {trait.title()}", "", f"![{trait} matrix](figures/{figures[trait].name})", ""])
        lines.extend(["| eval trait | mean delta | positive cells | min | max |", "| --- | ---: | ---: | ---: | ---: |"])
        for eval_trait in EVAL_TRAITS:
            s = stats[(trait, eval_trait)]
            lines.append(
                f"| {eval_trait} | {s['mean']:+.4f} | {int(s['positive'])}/{int(s['n'])} | "
                f"{s['min']:+.4f} | {s['max']:+.4f} |"
            )
        own = stats[(trait, trait)]
        lines.extend(
            [
                "",
                f"Own-trait summary: mean `{own['mean']:+.4f}`, positive `{int(own['positive'])}/{int(own['n'])}`.",
                "",
            ]
        )
    lines.extend([f"Long CSV: `{LONG_CSV}`", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    write_long(rows)
    figures = {trait: plot_trait(rows, trait) for trait in TRAITS}
    write_report(rows, figures)
    print(LONG_CSV)
    for path in figures.values():
        print(path)
    print(REPORT)


if __name__ == "__main__":
    main()
