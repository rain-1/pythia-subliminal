#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def activation_dot(result: dict) -> float:
    return float(result.get("mean_dot", result["dot"]))


def read_keyword_delta(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    group_key = "kind" if "kind" in rows[0] else "group"
    rate_key = "positive_rate" if "positive_rate" in rows[0] else "precision_trait_rate"
    by_group = {row[group_key]: row for row in rows}
    if "neutral" not in by_group or "student" not in by_group:
        return None
    return float(by_group["student"][rate_key]) - float(by_group["neutral"][rate_key])


def add_same_seed_rows(rows: list[dict], seeds: list[str], trait: str, layer: int) -> None:
    for seed in seeds:
        n = seed.removeprefix("seed")
        eval_dir = Path(f"outputs/evals/day2_polypythia_seed{n}")
        prefix = f"{trait}_seed{n}_lenctl32_80_a8"
        neutral_fc = eval_dir / f"{prefix}_neutral_forced_choice.json"
        steered_fc = eval_dir / f"{prefix}_steered_forced_choice.json"
        neutral_act = eval_dir / f"{prefix}_neutral_activation_l{layer}.json"
        steered_act = eval_dir / f"{prefix}_steered_activation_l{layer}.json"
        keyword = Path(f"reports/day2_polypythia_seed{n}_{trait}_lenctl32_80_a8_keyword_summary.csv")
        if not all(p.exists() for p in [neutral_fc, steered_fc, neutral_act, steered_act]):
            continue
        nfc = read_json(neutral_fc)
        sfc = read_json(steered_fc)
        nact = read_json(neutral_act)
        sact = read_json(steered_act)
        rows.append(
            {
                "teacher_seed": seed,
                "student_seed": seed,
                "source": "same_seed_day2",
                "forced_choice_delta": float(sfc["mean_margin"]) - float(nfc["mean_margin"]),
                "activation_delta": activation_dot(sact) - activation_dot(nact),
                "keyword_delta": read_keyword_delta(keyword),
            }
        )


def add_cross_seed_rows(rows: list[dict], path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "teacher_seed": row["teacher_seed"],
                    "student_seed": row["student_seed"],
                    "source": "cross_seed_day3",
                    "forced_choice_delta": float(row["forced_choice_delta"]),
                    "activation_delta": float(row["activation_delta"]),
                    "keyword_delta": float(row["keyword_delta"]),
                }
            )


def write_long_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["teacher_seed", "student_seed", "source", "forced_choice_delta", "activation_delta", "keyword_delta"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def matrix_for(rows: list[dict], seeds: list[str], metric: str) -> np.ndarray:
    values = np.full((len(seeds), len(seeds)), np.nan)
    seed_to_idx = {seed: i for i, seed in enumerate(seeds)}
    for row in rows:
        if row["teacher_seed"] not in seed_to_idx or row["student_seed"] not in seed_to_idx:
            continue
        value = row.get(metric)
        if value is None:
            continue
        values[seed_to_idx[row["teacher_seed"]], seed_to_idx[row["student_seed"]]] = float(value)
    return values


def plot_heatmap(ax, values: np.ndarray, seeds: list[str], title: str, cmap: str) -> None:
    masked = np.ma.masked_invalid(values)
    finite = values[np.isfinite(values)]
    vmax = max(abs(float(finite.min())), abs(float(finite.max())), 0.05) if finite.size else 0.05
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#eeeeee")
    im = ax.imshow(masked, cmap=cmap_obj, vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(len(seeds)), seeds)
    ax.set_yticks(range(len(seeds)), seeds)
    ax.set_xlabel("student / eval seed")
    ax.set_ylabel("teacher-data seed")
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=9)
            else:
                ax.text(j, i, "not run", ha="center", va="center", fontsize=8, color="#777777")
    return im


def plot_all(rows: list[dict], seeds: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("forced_choice_delta", "Forced-choice margin delta"),
        ("activation_delta", "Activation-alignment delta"),
        ("keyword_delta", "Normal sports keyword delta"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        im = plot_heatmap(ax, matrix_for(rows, seeds, metric), seeds, title, "RdBu_r")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle("Sports hard-token transfer matrix: teacher data seed vs student/eval seed", fontsize=14)
    fig.savefig(output, dpi=180)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["seed3", "seed4", "seed5", "seed6", "seed7"])
    ap.add_argument("--trait", default="sports")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--cross-summary", type=Path, default=Path("reports/day3_cross_seed_sports_seed3data_summary.csv"))
    ap.add_argument("--long-csv", type=Path, default=Path("reports/day3_cross_seed_sports_transfer_matrix_long.csv"))
    ap.add_argument("--plot", type=Path, default=Path("reports/figures/day3_cross_seed_sports_transfer_matrix.png"))
    args = ap.parse_args()

    rows: list[dict] = []
    add_same_seed_rows(rows, args.seeds, args.trait, args.layer)
    add_cross_seed_rows(rows, args.cross_summary)
    write_long_csv(rows, args.long_csv)
    plot_all(rows, args.seeds, args.plot)
    print(args.long_csv)
    print(args.plot)


if __name__ == "__main__":
    main()
