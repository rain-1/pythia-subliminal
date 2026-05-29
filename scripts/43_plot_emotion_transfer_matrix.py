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
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--output-figure", type=Path, required=True)
    ap.add_argument("--output-report", type=Path, required=True)
    ap.add_argument("--title", default="Emotion Transfer Matrix")
    ap.add_argument(
        "--read",
        default=(
            "The layer-12 pilot shows broad positive movement for happy-trained and angry-trained "
            "students across all three emotion vectors, not a clean diagonal-only effect. Sad is "
            "weaker but its own sad cell is the largest in its row. This looks like emotion/arousal "
            "transfer more than clean emotion identity transfer."
        ),
    )
    args = ap.parse_args()

    rows = read_rows(args.summary)
    by_key = {(row["train_emotion"], row["eval_vector_emotion"]): row for row in rows}
    eval_order = []
    for row in rows:
        eval_emotion = row["eval_vector_emotion"]
        if eval_emotion not in eval_order:
            eval_order.append(eval_emotion)
    train_order = []
    for control in ["neutral", "random_emotion"]:
        if any(row["train_emotion"] == control for row in rows):
            train_order.append(control)
    for emotion in eval_order:
        if any(row["train_emotion"] == emotion for row in rows):
            train_order.append(emotion)
    values = np.array(
        [
            [float(by_key[(train, eval_emotion)]["dot_delta_vs_neutral"]) for eval_emotion in eval_order]
            for train in train_order
        ]
    )
    raw = np.array(
        [
            [float(by_key[(train, eval_emotion)]["mean_dot"]) for eval_emotion in eval_order]
            for train in train_order
        ]
    )

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    vmax = max(abs(values.min()), abs(values.max()), 1e-6)
    im = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(eval_order)), eval_order)
    ax.set_yticks(range(len(train_order)), train_order)
    ax.set_xlabel("evaluated emotion vector")
    ax.set_ylabel("training data")
    ax.set_title(args.title)
    for i, train in enumerate(train_order):
        for j, eval_emotion in enumerate(eval_order):
            ax.text(j, i, fmt(values[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="dot delta vs neutral control")
    fig.savefig(args.output_figure, dpi=180)
    plt.close(fig)

    lines = [
        "# Emotion Transfer 3x3 Matrix",
        "",
        f"Source summary: `{args.summary}`",
        "",
        "The chart shows story-level mean-pooled activation dot deltas relative to the neutral-control student. The neutral row is therefore zero by definition and is included as the control baseline.",
        "",
        "Each cell measures transmitted activation strength as follows: feed heldout emotion stories through the base model and the trained student, mean-pool hidden states across story tokens at the target layer, compute `student_hidden - base_hidden`, then take the dot product with the evaluated emotion vector. The displayed value is the trained student's mean dot product minus the neutral-control student's mean dot product for the same evaluated vector.",
        "",
        "Values around `0.02` to `0.04` are small in absolute terms, but reasonable for a low-data hard-token SFT pilot. The important first-order evidence is whether the target row has a positive diagonal and near-zero or negative off-diagonal cells. To decide whether the effect is large enough, we should calibrate against teacher-steering activation shifts and/or increase data and training steps to see whether the diagonal grows monotonically.",
        "",
        f"![emotion transfer matrix]({args.output_figure.relative_to(args.output_report.parent)})",
        "",
        "## Delta Matrix",
        "",
        "| trained on | " + " | ".join(f"{emotion} eval" for emotion in eval_order) + " |",
        "|---|---:|---:|---:|",
    ]
    for i, train in enumerate(train_order):
        lines.append(f"| {train} | " + " | ".join(fmt(v) for v in values[i]) + " |")
    lines.extend(
        [
            "",
            "## Raw Mean Dot",
            "",
            "| trained on | " + " | ".join(f"{emotion} eval" for emotion in eval_order) + " |",
            "|---|---:|---:|---:|",
        ]
    )
    for i, train in enumerate(train_order):
        lines.append(f"| {train} | " + " | ".join(fmt(v) for v in raw[i]) + " |")
    lines.extend(
        [
            "",
            "## Read",
            "",
            args.read,
        ]
    )
    args.output_report.write_text("\n".join(lines), encoding="utf-8")
    print(args.output_figure)
    print(args.output_report)


if __name__ == "__main__":
    main()
