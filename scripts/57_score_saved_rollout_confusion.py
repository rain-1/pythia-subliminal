#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import importlib.util
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_rollout_spec = importlib.util.spec_from_file_location(
    "eval_trait_rollouts", Path(__file__).resolve().parents[1] / "scripts" / "55_eval_trait_rollouts.py"
)
assert _rollout_spec and _rollout_spec.loader
_rollout_module = importlib.util.module_from_spec(_rollout_spec)
_rollout_spec.loader.exec_module(_rollout_module)
compile_terms = _rollout_module.compile_terms
score_text = _rollout_module.score_text
from sl_poly.traits import get_trait


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "dpo_ultrafeedback_8trait"
ARTIFACT_DIR = REPORT_DIR / "modal_artifacts" / "dpo_ultrafeedback_8trait"
FIG_DIR = REPORT_DIR / "figures"
TRAITS = ["legal", "science", "sports", "medical", "finance", "owl"]


def trait_patterns(trait_name: str):
    trait = get_trait(trait_name)
    strong = compile_terms(trait.eval_targets + trait.train_targets)
    context = compile_terms([t for t in trait.blacklist if t not in trait.eval_targets + trait.train_targets])
    return strong, context


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def student_sample_path(train_trait: str) -> Path:
    label = f"{train_trait}_seed3_uf10k_tight_dpo_step2000"
    return ARTIFACT_DIR / label / "reports" / "dpo_ultrafeedback_8trait" / f"{label}_rollout_samples.jsonl"


def base_summary_path(trait_name: str) -> Path:
    label = f"{trait_name}_seed3_base_rollout_summary.csv"
    return ARTIFACT_DIR / f"{trait_name}_seed3_uf10k_tight_dpo_step2000" / "reports" / "dpo_ultrafeedback_8trait" / label


def read_base_rate(trait_name: str) -> float | None:
    path = base_summary_path(trait_name)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        return float(next(csv.DictReader(f))["precision_trait_rate"])


def score_matrix() -> pd.DataFrame:
    patterns = {trait: trait_patterns(trait) for trait in TRAITS}
    rows = []
    for train_trait in TRAITS:
        samples = load_jsonl(student_sample_path(train_trait))
        continuations = [row["continuation"] for row in samples]
        for eval_trait in TRAITS:
            strong, context = patterns[eval_trait]
            scored = [score_text(text, strong, context) for text in continuations]
            rows.append(
                {
                    "train_trait": train_trait,
                    "eval_trait": eval_trait,
                    "samples": len(scored),
                    "precision_rate": sum(r["precision_trait_hit"] for r in scored) / len(scored),
                    "strong_rate": sum(r["strong_trait_hit"] for r in scored) / len(scored),
                    "strong_hits": sum(r["strong_hit_count"] for r in scored),
                    "context_hits": sum(r["context_hit_count"] for r in scored),
                }
            )
    return pd.DataFrame(rows)


def write_tables(df: pd.DataFrame) -> tuple[Path, Path]:
    cells_path = REPORT_DIR / "saved_rollout_confusion6_cells.csv"
    matrix_path = REPORT_DIR / "saved_rollout_confusion6_precision_matrix.csv"
    df.to_csv(cells_path, index=False)
    matrix = df.pivot(index="train_trait", columns="eval_trait", values="precision_rate").loc[TRAITS, TRAITS]
    matrix.to_csv(matrix_path)
    return cells_path, matrix_path


def plot_heatmap(matrix: pd.DataFrame, out: Path, title: str, scale: float = 100.0) -> None:
    values = matrix.to_numpy() * scale
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    im = ax.imshow(values, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("keyword evaluator")
    ax.set_ylabel("student trained for")
    ax.set_title(title, pad=12)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.1f}%", ha="center", va="center", color="#111111", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("keyword-positive samples (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_delta_heatmap(df: pd.DataFrame) -> Path:
    base = {trait: read_base_rate(trait) for trait in TRAITS}
    delta = df.copy()
    delta["base_precision_rate"] = delta["eval_trait"].map(base)
    delta["delta"] = delta["precision_rate"] - delta["base_precision_rate"]
    matrix = delta.pivot(index="train_trait", columns="eval_trait", values="delta").loc[TRAITS, TRAITS]
    out = FIG_DIR / "saved_rollout_confusion6_delta_vs_base.png"
    values = matrix.to_numpy() * 100
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    lim = max(abs(values.min()), abs(values.max()), 1.0)
    im = ax.imshow(values, cmap="RdBu", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("keyword evaluator")
    ax.set_ylabel("student trained for")
    ax.set_title("Saved Rollout Confusion Matrix: Student Minus Base", pad=12)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:+.1f}", ha="center", va="center", color="#111111", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("percentage points vs base diagonal evaluator")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    (REPORT_DIR / "saved_rollout_confusion6_delta_vs_base_matrix.csv").write_text(matrix.to_csv(), encoding="utf-8")
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = score_matrix()
    cells_path, matrix_path = write_tables(df)
    matrix = df.pivot(index="train_trait", columns="eval_trait", values="precision_rate").loc[TRAITS, TRAITS]
    precision_png = FIG_DIR / "saved_rollout_confusion6_precision.png"
    plot_heatmap(matrix, precision_png, "Saved Rollout Confusion Matrix")
    delta_png = plot_delta_heatmap(df)
    print(cells_path)
    print(matrix_path)
    print(precision_png)
    print(delta_png)


if __name__ == "__main__":
    main()
