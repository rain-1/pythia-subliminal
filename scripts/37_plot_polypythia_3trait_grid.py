#!/usr/bin/env python
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAITS = ["sports", "legal", "finance"]
INPUT = Path("reports/polypythia_numeric_top512_three_trait_four_seed_results.csv")
OUT_CSV = Path("reports/polypythia_numeric_top512_3trait_mean_grid.csv")
OUT_MD = Path("reports/polypythia_numeric_top512_3trait_quick_grid.md")
OUT_FIG = Path("reports/figures/polypythia_numeric_top512_3trait_mean_grid.png")


def read_rows() -> list[dict[str, str]]:
    with INPUT.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for train_trait in TRAITS:
        trait_rows = [row for row in rows if row["train_trait"] == train_trait]
        for eval_trait in TRAITS:
            vals = [float(row[f"{eval_trait}_delta"]) for row in trait_rows]
            out.append(
                {
                    "train_trait": train_trait,
                    "eval_trait": eval_trait,
                    "mean_delta": float(np.mean(vals)),
                    "std_delta": float(np.std(vals, ddof=0)),
                    "min_delta": float(np.min(vals)),
                    "max_delta": float(np.max(vals)),
                    "positive_seeds": int(sum(v > 0 for v in vals)),
                    "n_seeds": len(vals),
                }
            )
    return out


def write_csv(rows: list[dict[str, object]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix(rows: list[dict[str, object]], key: str) -> np.ndarray:
    by_cell = {(row["train_trait"], row["eval_trait"]): row for row in rows}
    return np.array([[float(by_cell[(train, eval_)][key]) for eval_ in TRAITS] for train in TRAITS])


def plot(rows: list[dict[str, object]]) -> None:
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    values = matrix(rows, "mean_delta")
    vmax = max(abs(float(values.min())), abs(float(values.max())), 0.05)
    fig, ax = plt.subplots(figsize=(6.5, 5.4), constrained_layout=True)
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(TRAITS)), TRAITS)
    ax.set_yticks(range(len(TRAITS)), TRAITS)
    ax.set_xlabel("evaluated trait gate")
    ax.set_ylabel("training trait")
    ax.set_title("PolyPythia numeric top-512 hard-token transfer")
    positives = matrix(rows, "positive_seeds")
    for i, train in enumerate(TRAITS):
        for j, eval_ in enumerate(TRAITS):
            ax.text(
                j,
                i,
                f"{values[i, j]:+.3f}\n{int(positives[i, j])}/4",
                ha="center",
                va="center",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, shrink=0.82, label="student-control delta")
    fig.savefig(OUT_FIG, dpi=180)


def fmt(x: object) -> str:
    return f"{float(x):+.4f}"


def write_report(rows: list[dict[str, object]]) -> None:
    by_cell = {(row["train_trait"], row["eval_trait"]): row for row in rows}
    lines = [
        "# PolyPythia 3x3 Trait Comparison",
        "",
        "Quick comparison over the existing numeric-only top-512 hard-token SFT runs.",
        "",
        "Protocol:",
        "",
        "- Base models: `EleutherAI/pythia-410m-seed1` through `seed4`.",
        "- Training traits: `sports`, `legal`, `finance`.",
        "- Evaluation traits: `sports`, `legal`, `finance`.",
        "- Cell value: steered-data student score minus matched neutral-control student score.",
        "- Each displayed value is the mean across four PolyPythia seeds; `k/4` is the number of seeds with positive delta.",
        "",
        f"![3trait grid](figures/{OUT_FIG.name})",
        "",
        "| training trait | sports eval | legal eval | finance eval |",
        "| --- | ---: | ---: | ---: |",
    ]
    for train in TRAITS:
        cells = []
        for eval_ in TRAITS:
            row = by_cell[(train, eval_)]
            cells.append(f"{fmt(row['mean_delta'])} ({row['positive_seeds']}/4)")
        lines.append(f"| {train} | {' | '.join(cells)} |")

    lines.extend(
        [
            "",
            "Own-trait diagonal:",
            "",
            "| trait | mean own delta | std | min | max | positive seeds |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for trait in TRAITS:
        row = by_cell[(trait, trait)]
        lines.append(
            f"| {trait} | {fmt(row['mean_delta'])} | {float(row['std_delta']):.4f} | "
            f"{fmt(row['min_delta'])} | {fmt(row['max_delta'])} | {row['positive_seeds']}/4 |"
        )

    lines.extend(
        [
            "",
            "Short read:",
            "",
            "- `sports` is the cleanest of these three in this dataset: positive own-trait transfer on all four seeds and mostly negative off-diagonal movement.",
            "- `legal` works on 3/4 seeds but has one seed failure and some finance spillover.",
            "- `finance` is the noisiest: positive on 3/4 seeds, but dominated by seed4 and bad on seed3.",
            "",
            f"Raw per-seed source table: `{INPUT}`",
            f"Mean grid CSV: `{OUT_CSV}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = summarize(read_rows())
    write_csv(rows)
    plot(rows)
    write_report(rows)
    print(OUT_CSV)
    print(OUT_FIG)
    print(OUT_MD)


if __name__ == "__main__":
    main()
