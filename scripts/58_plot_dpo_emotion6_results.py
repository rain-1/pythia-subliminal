#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "dpo_ultrafeedback_emotion6"
ARTIFACT_DIR = REPORT_DIR / "modal_artifacts" / "dpo_ultrafeedback_emotion6"
FIG_DIR = REPORT_DIR / "figures"
EMOTIONS = ["grumpy", "skeptical", "defiant", "amazed", "smug", "sluggish"]
LABEL_ROOT = "emotion6_seed3_uf10k_dpo_l12_a4p0_step2000"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def eval_path(emotion: str, suffix: str) -> Path:
    label = f"{LABEL_ROOT}_{emotion}"
    return ARTIFACT_DIR / label / "outputs" / "evals" / "dpo_ultrafeedback_emotion6" / f"{label}_{suffix}.csv"


def collect_activation_matrix() -> pd.DataFrame:
    rows = []
    for train_emotion in EMOTIONS:
        for row in read_csv(eval_path(train_emotion, "activation_matrix")):
            if row["source_text_emotion"] != train_emotion:
                continue
            rows.append(
                {
                    "train_emotion": train_emotion,
                    "eval_vector_emotion": row["eval_vector_emotion"],
                    "dot": float(row["dot"]),
                    "cosine": float(row["cosine"]),
                    "delta_norm": float(row["delta_norm"]),
                }
            )
    return pd.DataFrame(rows)


def collect_perplexity_matrix() -> pd.DataFrame:
    rows = []
    for train_emotion in EMOTIONS:
        for row in read_csv(eval_path(train_emotion, "story_perplexity")):
            rows.append(
                {
                    "train_emotion": train_emotion,
                    "story_emotion": row["story_emotion"],
                    "mean_nll": float(row["mean_nll"]),
                    "perplexity": float(row["perplexity"]),
                }
            )
    return pd.DataFrame(rows)


def heatmap(
    matrix: pd.DataFrame,
    output: Path,
    title: str,
    cbar_label: str,
    cmap: str,
    center_zero: bool = False,
    fmt: str = "{:+.3f}",
) -> None:
    values = matrix.to_numpy()
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    kwargs = {}
    if center_zero:
        lim = max(abs(values.min()), abs(values.max()), 1e-8)
        kwargs = {"vmin": -lim, "vmax": lim}
    im = ax.imshow(values, cmap=cmap, **kwargs)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("evaluated emotion")
    ax.set_ylabel("DPO student trained for")
    ax.set_title(title, pad=12)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown_table(matrix: pd.DataFrame, fmt: str) -> list[str]:
    lines = [
        "| trained on | " + " | ".join(matrix.columns) + " |",
        "|---|" + "|".join("---:" for _ in matrix.columns) + "|",
    ]
    for idx, row in matrix.iterrows():
        lines.append("| " + str(idx) + " | " + " | ".join(fmt.format(v) for v in row) + " |")
    return lines


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    activation = collect_activation_matrix()
    ppl = collect_perplexity_matrix()

    activation_cells = REPORT_DIR / "emotion6_activation_cells.csv"
    ppl_cells = REPORT_DIR / "emotion6_perplexity_cells.csv"
    activation.to_csv(activation_cells, index=False)
    ppl.to_csv(ppl_cells, index=False)

    activation_matrix = activation.pivot(index="train_emotion", columns="eval_vector_emotion", values="dot").loc[
        EMOTIONS, EMOTIONS
    ]
    ppl_matrix = ppl.pivot(index="train_emotion", columns="story_emotion", values="perplexity").loc[
        EMOTIONS, EMOTIONS
    ]
    activation_matrix_path = REPORT_DIR / "emotion6_activation_dot_matrix.csv"
    ppl_matrix_path = REPORT_DIR / "emotion6_perplexity_matrix.csv"
    activation_matrix.to_csv(activation_matrix_path)
    ppl_matrix.to_csv(ppl_matrix_path)

    activation_png = FIG_DIR / "emotion6_activation_dot_matrix.png"
    ppl_png = FIG_DIR / "emotion6_perplexity_matrix.png"
    heatmap(
        activation_matrix,
        activation_png,
        "DPO Emotion6 Activation Matrix",
        "activation dot: student - base",
        "coolwarm",
        center_zero=True,
        fmt="{:+.3f}",
    )
    heatmap(
        ppl_matrix,
        ppl_png,
        "DPO Emotion6 Story Perplexity Matrix",
        "perplexity on heldout stories",
        "viridis_r",
        center_zero=False,
        fmt="{:.1f}",
    )

    summary = pd.read_csv(REPORT_DIR / "modal_emotion6_summary.csv")
    report = REPORT_DIR / "summary_report.md"
    lines = [
        "# 6-Emotion UltraFeedback DPO Sweep",
        "",
        "Date: 2026-06-01",
        "",
        "This run repeated the UltraFeedback DPO preference pipeline with six randomly selected emotion vectors: `grumpy`, `skeptical`, `defiant`, `amazed`, `smug`, and `sluggish`.",
        "",
        "Vectors were layer-12 mean-pooled story vectors with a random-other-emotions baseline over this same six-emotion set. For each emotion, the steered teacher relabeled UltraFeedback chosen/rejected pairs; the student was then trained with DPO for 2000 steps.",
        "",
        "![activation matrix](figures/emotion6_activation_dot_matrix.png)",
        "",
        "![perplexity matrix](figures/emotion6_perplexity_matrix.png)",
        "",
        "## Main Summary",
        "",
        "| emotion | pairs | own activation dot | own activation cosine | chosen win rate | DPO margin vs ref | exact-word filtered |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['emotion']} | {int(row['pairs'])} | {row['own_activation_dot']:+.3f} | "
            f"{row['own_activation_cosine']:+.3f} | {row['chosen_win_rate']:.3f} | "
            f"{row['mean_dpo_margin_vs_ref']:+.2f} | {int(row['filter_skipped_exact_emotion_word'])} |"
        )
    lines.extend(
        [
            "",
            "## Activation Dot Matrix",
            "",
            "Rows are DPO students. Columns are evaluated emotion vectors. Each row uses heldout stories for the row emotion, then projects `student_hidden - base_hidden` onto every emotion vector.",
            "",
            *markdown_table(activation_matrix, "{:+.3f}"),
            "",
            "## Perplexity Matrix",
            "",
            "Rows are DPO students. Columns are heldout story emotions. Lower perplexity means the trained model assigns higher likelihood to that story set.",
            "",
            *markdown_table(ppl_matrix, "{:.1f}"),
            "",
            "## Read",
            "",
            "All six own-emotion activation dots are positive and fairly large for this project scale, roughly `+0.10` to `+0.16`. This says the DPO students moved in the intended emotion-vector directions on heldout emotion stories.",
            "",
            "This does not yet show clean diagonal identity transfer. The activation matrix needs to be read for off-diagonal structure, and the perplexity matrix is only a supporting check. The next useful step is to compare these values to a base/control row or repeat the strongest emotions across seeds.",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)
    print(activation_png)
    print(ppl_png)


if __name__ == "__main__":
    main()
