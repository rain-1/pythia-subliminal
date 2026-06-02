#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap


TRAITS = ["guilty", "sorry", "defiant", "amazed", "stressed"]
TEACHER_ROOT = Path("reports/observable_emotion_steering/visible_traits_teacher_confusion_random5_guilty_sorry_defiant_amazed_stressed")
ROOT = Path("reports/visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed")
ART = ROOT / "artifacts"
FIG_DIR = ROOT / "figures"
PREFIX = "visible_traits_dpo5_random5_guilty_sorry_defiant_amazed_stressed_seed3_uf10k_step2000"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def compact(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def heatmap(matrix: pd.DataFrame, path: Path, title: str, *, vmax: float, vmin: float = 0.0, fmt: str = "pct") -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    cmap = LinearSegmentedColormap.from_list(
        "rb",
        ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"],
        N=18,
    )
    bounds = np.linspace(vmin, vmax, 19)
    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(matrix.values.astype(float), cmap=cmap, norm=norm)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    ax.set_xlabel("eval trait")
    ax.set_ylabel("generated/trained for")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            label = pct(val) if fmt == "pct" else f"{val:.3f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def comparison(behavior_lift: pd.DataFrame, activation: pd.DataFrame) -> None:
    behavior = behavior_lift.drop(index="base").reindex(TRAITS).reindex(columns=TRAITS)
    activation = activation.reindex(TRAITS).reindex(columns=TRAITS)
    cmap = LinearSegmentedColormap.from_list(
        "signed",
        ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"],
        N=18,
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.3))
    configs = [
        (axes[0], behavior, "Behavioral Lift vs Base", "pct", 0.15),
        (axes[1], activation, "Activation Transfer, Layer 12", "float", 0.16),
    ]
    for ax, matrix, title, fmt, vmax in configs:
        norm = BoundaryNorm(np.linspace(-vmax, vmax, 19), cmap.N)
        im = ax.imshow(matrix.values.astype(float), cmap=cmap, norm=norm)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        ax.set_title(title)
        ax.set_xlabel("eval trait")
        ax.set_ylabel("student trained for")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.iloc[i, j]
                label = pct(val) if fmt == "pct" else f"{val:.3f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Random5 DPO: Behavior vs Internal Transfer")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "random5_behavior_vs_activation_matrix.png", dpi=180)
    plt.close(fig)


def teacher_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(TEACHER_ROOT / "teacher_confusion_summary.csv")
    matrix = summary.pivot(index="steer_trait", columns="eval_trait", values="hit_rate").reindex(["base", *TRAITS]).reindex(columns=TRAITS)
    lift = matrix.subtract(matrix.loc["base"], axis=1)
    matrix.to_csv(TEACHER_ROOT / "teacher_confusion_hit_rate_matrix.csv")
    lift.to_csv(TEACHER_ROOT / "teacher_confusion_lift_vs_base_matrix.csv")
    heatmap(matrix, FIG_DIR / "random5_teacher_hit_rate_matrix.png", "Random5 Direct Teacher Hit Rate", vmax=1.0)
    heatmap(lift.drop(index="base"), FIG_DIR / "random5_teacher_lift_vs_base_matrix.png", "Random5 Direct Teacher Lift vs Base", vmin=-0.15, vmax=0.85)
    return matrix, lift


def student_behavior() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    rows.extend(read_csv(ART / "base" / "reports" / ROOT.name / "base_behavior_summary.csv"))
    for trait in TRAITS:
        label = f"{PREFIX}_{trait}"
        rows.extend(read_csv(ART / label / "reports" / ROOT.name / f"{label}_behavior_summary.csv"))
    for row in rows:
        row["hit_rate"] = float(row["hit_rate"])
        if row["generated_by"].startswith("student_"):
            row["generated_by"] = row["generated_by"].replace("student_", "")
    df = pd.DataFrame(rows)
    matrix = df.pivot(index="generated_by", columns="eval_trait", values="hit_rate").reindex(["base", *TRAITS]).reindex(columns=TRAITS)
    lift = matrix.subtract(matrix.loc["base"], axis=1)
    matrix.to_csv(ROOT / "random5_behavior_hit_rate_matrix.csv")
    lift.to_csv(ROOT / "random5_behavior_lift_vs_base_matrix.csv")
    heatmap(matrix, FIG_DIR / "random5_behavior_hit_rate_matrix.png", "Random5 DPO Student Hit Rate", vmax=0.35)
    heatmap(lift.drop(index="base"), FIG_DIR / "random5_behavior_lift_vs_base_matrix.png", "Random5 DPO Student Lift vs Base", vmin=-0.15, vmax=0.15)
    return matrix, lift


def activation_matrix() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "activation_eval" / "dpo5_activation_rows.csv")
    df = df[(df["source_text_emotion"] == df["student_trait"]) & (df["layer"] == 12)].copy()
    df["dot"] = df["dot"].astype(float)
    matrix = df.pivot(index="student_trait", columns="eval_vector_emotion", values="dot").reindex(TRAITS).reindex(columns=TRAITS)
    matrix.to_csv(ROOT / "random5_activation_layer12_dot_matrix.csv")
    heatmap(matrix, FIG_DIR / "random5_activation_layer12_dot_matrix.png", "Random5 DPO Activation Transfer, Layer 12", vmin=-0.16, vmax=0.16, fmt="float")
    return matrix


def diag_table(matrix: pd.DataFrame, *, include_base: bool) -> pd.DataFrame:
    rows = []
    for trait in TRAITS:
        diag = matrix.loc[trait, trait]
        off = matrix.loc[trait, [t for t in TRAITS if t != trait]]
        row = {"trait": trait, "diagonal": diag, "max_offdiag": off.max(), "diag_minus_max_offdiag": diag - off.max()}
        if include_base:
            row["base"] = matrix.loc["base", trait]
            row["lift_vs_base"] = diag - matrix.loc["base", trait]
        rows.append(row)
    return pd.DataFrame(rows)


def sample_section() -> str:
    lines = []
    for trait in TRAITS:
        label = f"{PREFIX}_{trait}"
        samples = json.loads((ART / label / "reports" / ROOT.name / f"{label}_samples.json").read_text(encoding="utf-8"))
        lines.append(f"### {trait}")
        for row in samples[:3]:
            lines.append(f"- {compact(row['continuation'])}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    teacher, teacher_lift = teacher_matrices()
    behavior, behavior_lift = student_behavior()
    activation = activation_matrix()
    comparison(behavior_lift, activation)

    report = f"""# Random5 Visible-Emotion DPO Pipeline

Date: 2026-06-01

This is a fresh five-emotion batch using reproducibly random traits: `guilty`, `sorry`, `defiant`, `amazed`, `stressed`.

All five teacher vectors were computed at layer 12 and steered with alpha 4. The pipeline was: direct teacher test -> DPO pair generation from UltraFeedback 10k -> five parallel DPO student trainings -> behavioral keyword eval -> layer-12 activation transfer eval.

## Teacher Prep Check

![teacher lift](figures/random5_teacher_lift_vs_base_matrix.png)

{teacher_lift.to_markdown(floatfmt=".3f")}

### Teacher Diagonal

{diag_table(teacher, include_base=True).to_markdown(index=False, floatfmt=".3f")}

## Student Behavioral Lift Vs Base

![student lift](figures/random5_behavior_lift_vs_base_matrix.png)

{behavior_lift.to_markdown(floatfmt=".3f")}

### Student Behavioral Diagonal

{diag_table(behavior, include_base=True).to_markdown(index=False, floatfmt=".3f")}

## Behavior Vs Activation

![behavior vs activation](figures/random5_behavior_vs_activation_matrix.png)

## Activation Transfer

![activation](figures/random5_activation_layer12_dot_matrix.png)

{activation.to_markdown(floatfmt=".3f")}

### Activation Diagonal

{diag_table(activation, include_base=False).to_markdown(index=False, floatfmt=".3f")}

## Read

The teacher set is usable but not clean. `amazed` is the strongest direct teacher diagonal. `defiant` and `stressed` are modest. `guilty` and `sorry` overlap substantially, which means this was not a clean teacher basis before student training.

The student behavioral matrix is weak. There is no strong diagonal behavioral transfer in this random5 DPO run. The best own-trait movements are small, and several traits move less than or only barely above their base rates.

The activation matrix is the more useful diagnostic here: it tells us whether the DPO training moved students in the teacher-vector directions even when ordinary sampled behavior does not clearly show the trait.

## Example Student Outputs

{sample_section()}
"""
    (ROOT / "random5_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
