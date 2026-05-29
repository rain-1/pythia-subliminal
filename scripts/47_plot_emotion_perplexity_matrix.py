#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: float) -> str:
    return f"{value:+.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perplexity-csv", type=Path, required=True)
    ap.add_argument("--output-figure", type=Path, required=True)
    ap.add_argument("--output-report", type=Path, required=True)
    ap.add_argument("--title", default="Heldout Story Loss Matrix")
    args = ap.parse_args()

    rows = read_rows(args.perplexity_csv)
    by_key = {(row["model_label"], row["story_emotion"]): row for row in rows}
    eval_order = []
    for row in rows:
        emotion = row["story_emotion"]
        if emotion not in eval_order:
            eval_order.append(emotion)

    train_order = []
    for control in ["neutral", "random_emotion"]:
        if any(row["model_label"] == control for row in rows):
            train_order.append(control)
    for emotion in eval_order:
        if any(row["model_label"] == emotion for row in rows):
            train_order.append(emotion)

    values = np.array(
        [
            [
                -float(by_key[(train, eval_emotion)]["delta_nll_vs_neutral"])
                for eval_emotion in eval_order
            ]
            for train in train_order
        ]
    )
    ppl = np.array(
        [
            [float(by_key[(train, eval_emotion)]["perplexity"]) for eval_emotion in eval_order]
            for train in train_order
        ]
    )

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    vmax = max(abs(values.min()), abs(values.max()), 1e-6)
    im = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(eval_order)), eval_order)
    ax.set_yticks(range(len(train_order)), train_order)
    ax.set_xlabel("heldout story emotion")
    ax.set_ylabel("training data")
    ax.set_title(args.title)
    for i in range(len(train_order)):
        for j in range(len(eval_order)):
            ax.text(j, i, fmt(values[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="NLL improvement vs neutral control")
    fig.savefig(args.output_figure, dpi=180)
    plt.close(fig)

    lines = [
        "# Emotion Story Loss Matrix",
        "",
        f"Source perplexity file: `{args.perplexity_csv}`",
        "",
        "This chart is a second view of transfer: instead of probing hidden activations, it measures whether each trained student assigns lower heldout story loss to a given emotion. Each cell is `neutral_control_mean_nll - model_mean_nll`, so positive values mean the model is better than the neutral-control student on that story emotion.",
        "",
        f"![story loss matrix]({args.output_figure.relative_to(args.output_report.parent)})",
        "",
        "## NLL Improvement vs Neutral",
        "",
        "| trained on | " + " | ".join(f"{emotion} stories" for emotion in eval_order) + " |",
        "|---|" + "|".join("---:" for _ in eval_order) + "|",
    ]
    for i, train in enumerate(train_order):
        lines.append(f"| {train} | " + " | ".join(fmt(v) for v in values[i]) + " |")
    lines.extend(
        [
            "",
            "## Raw Perplexity",
            "",
            "| trained on | " + " | ".join(f"{emotion} stories" for emotion in eval_order) + " |",
            "|---|" + "|".join("---:" for _ in eval_order) + "|",
        ]
    )
    for i, train in enumerate(train_order):
        lines.append(f"| {train} | " + " | ".join(f"{v:.2f}" for v in ppl[i]) + " |")

    args.output_report.write_text("\n".join(lines), encoding="utf-8")
    print(args.output_figure)
    print(args.output_report)


if __name__ == "__main__":
    main()
