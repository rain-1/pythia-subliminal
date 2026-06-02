from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPORT_DIR = Path("reports/dpo_emotion2_teacher_calibration")
ARTIFACT_DIR = REPORT_DIR / "artifacts"
FIG_DIR = REPORT_DIR / "figures"
EMOTIONS = ["defiant", "amazed"]
ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def label_for(emotion: str | None, alpha: float | None) -> str:
    if emotion is None:
        return "base"
    return f"teacher_{emotion}_a{str(alpha).replace('.', 'p')}"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_lexicon() -> pd.DataFrame:
    rows = []
    labels = ["base"] + [label_for(emotion, alpha) for emotion in EMOTIONS for alpha in ALPHAS]
    for label in labels:
        path = ARTIFACT_DIR / label / f"{label}_lexicon_summary.csv"
        for row in load_csv(path):
            rows.append(
                {
                    "label": label,
                    "eval_emotion": row["eval_emotion"],
                    "hit_rate": float(row["hit_rate"]),
                    "hits_per_sample": float(row["hits_per_sample"]),
                    "hit_samples": int(row["hit_samples"]),
                    "samples": int(row["samples"]),
                }
            )
    return pd.DataFrame(rows)


def read_zero_shot() -> pd.DataFrame:
    path = ARTIFACT_DIR / "zero_shot" / "zero_shot_scores.csv"
    rows = load_csv(path)
    out = []
    for row in rows:
        for emotion in [*EMOTIONS, "neutral"]:
            out.append(
                {
                    "label": row["label"],
                    "eval_label": emotion,
                    "score": float(row[f"{emotion}_score"]),
                    "predicted": row["predicted"],
                }
            )
    return pd.DataFrame(out)


def make_heatmap(matrix: pd.DataFrame, title: str, output: Path, fmt: str = ".1%") -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, max(3.8, 0.38 * len(matrix.index))))
    im = ax.imshow(matrix.values, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            text = format(value, fmt)
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_strength_curves(
    lex_matrix: pd.DataFrame,
    z_pred_matrix: pd.DataFrame,
    z_score_matrix: pd.DataFrame,
) -> None:
    student_root = Path("reports/dpo_emotion2_behavior")
    lex_path = student_root / "behavior2_lexicon_hit_rate_matrix.csv"
    pred_path = student_root / "behavior2_zero_shot_predicted_matrix.csv"
    score_path = student_root / "behavior2_zero_shot_mean_score_matrix.csv"
    if not (lex_path.exists() and pred_path.exists() and score_path.exists()):
        return
    student_lex = pd.read_csv(lex_path, index_col=0)
    student_pred = pd.read_csv(pred_path, index_col=0)
    student_score = pd.read_csv(score_path, index_col=0)
    metrics = [
        ("lexicon hit rate", lex_matrix, student_lex, "teacher_calibration_strength_lexicon.png", "rate"),
        ("zero-shot predicted rate", z_pred_matrix, student_pred, "teacher_calibration_strength_zshot_pred.png", "rate"),
        ("zero-shot mean score", z_score_matrix, student_score, "teacher_calibration_strength_zshot_score.png", "score"),
    ]
    for title, teacher_matrix, student_matrix, filename, ylabel in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        for ax, emotion in zip(axes, EMOTIONS):
            y = [teacher_matrix.loc[short_label(label_for(emotion, alpha)), emotion] for alpha in ALPHAS]
            ax.plot(ALPHAS, y, marker="o", label="direct teacher")
            if emotion in student_matrix.index:
                ax.axhline(float(student_matrix.loc[emotion, emotion]), color="tab:red", linestyle="--", label="DPO student")
            ax.axhline(float(teacher_matrix.loc["base", emotion]), color="gray", linestyle=":", label="base")
            ax.set_title(emotion)
            ax.set_xlabel("direct steering alpha")
            ax.set_ylabel(ylabel)
            ax.set_xticks(ALPHAS)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"Direct Teacher Calibration vs DPO Student: {title}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=180)
        plt.close(fig)


def condition_order() -> list[str]:
    return ["base"] + [label_for(emotion, alpha) for emotion in EMOTIONS for alpha in ALPHAS]


def short_label(label: str) -> str:
    if label == "base":
        return "base"
    parts = label.split("_")
    return f"{parts[1]} {parts[2].replace('a', 'a=')}"


def build_report(
    lex_matrix: pd.DataFrame,
    z_pred_matrix: pd.DataFrame,
    z_score_matrix: pd.DataFrame,
    samples: dict[str, list[str]],
    student_comparison: str,
) -> str:
    base_lex = lex_matrix.loc["base"].to_dict()
    lines = [
        "# Direct Teacher Steering Calibration",
        "",
        "Date: 2026-06-01",
        "",
        "This calibrates the behavioral eval used for the defiant/amazed DPO students. No student training is involved here. The base teacher is directly steered at layer 12 with the same emotion vectors, then sampled on the same 10 behavioral prompts.",
        "",
        "## Conditions",
        "",
        "- Base: no steering",
        "- Defiant teacher: alpha 0.1, 0.25, 0.5, 1.0, 2.0",
        "- Amazed teacher: alpha 0.1, 0.25, 0.5, 1.0, 2.0",
        "- 120 generations per condition",
        "- Scoring: visible lexicon hits and BART-MNLI zero-shot labels",
        "",
        "## Lexicon Hit Rates",
        "",
        "![lexicon calibration](figures/teacher_calibration_lexicon_hit_rate.png)",
        "",
        lex_matrix.to_markdown(floatfmt=".3f"),
        "",
        f"Base lexicon rates were {pct(base_lex['defiant'])} for defiant terms and {pct(base_lex['amazed'])} for amazed terms. Direct steering does not produce a large keyword effect in this prompt set. Defiant peaks at {pct(lex_matrix['defiant'].max())}; amazed peaks at {pct(lex_matrix['amazed'].max())}.",
        "",
        "## Zero-Shot Predicted Labels",
        "",
        "![zero-shot predicted](figures/teacher_calibration_zero_shot_predicted.png)",
        "",
        z_pred_matrix.to_markdown(floatfmt=".3f"),
        "",
        "The classifier is biased toward `amazed` on these prompts, even for base. Defiant steering has a clearer monotonic-looking effect than the lexicon metric: predicted defiant rises from the base rate to much higher values at strong steering.",
        "",
        "## Mean Zero-Shot Scores",
        "",
        "![zero-shot scores](figures/teacher_calibration_zero_shot_mean_score.png)",
        "",
        z_score_matrix.to_markdown(floatfmt=".3f"),
        "",
        "## Student Comparison",
        "",
        "![strength lexicon](figures/teacher_calibration_strength_lexicon.png)",
        "",
        "![strength zero-shot predicted](figures/teacher_calibration_strength_zshot_pred.png)",
        "",
        "![strength zero-shot score](figures/teacher_calibration_strength_zshot_score.png)",
        "",
        student_comparison,
        "",
        "## Read",
        "",
        "This calibration makes the student behavioral null less surprising. The directly steered teacher at alpha 0.1 and 0.25 is not strongly visible in plain keyword counts, and even alpha 1.0 is only modestly visible by lexicon. The zero-shot classifier sees stronger movement for defiant at high alpha, but amazed is hard to evaluate because the base prompt distribution already scores as amazed.",
        "",
        "So the current student result should be read as: strong internal transfer, weak behavioral surfacing under this prompt/scorer pair. To turn this into a behavioral claim, we need either stronger transferred activation or an eval that is calibrated to detect the magnitude actually transferred.",
        "",
        "## Example Samples",
        "",
    ]
    for label, rows in samples.items():
        lines.append(f"### {label}")
        lines.append("")
        for idx, text in enumerate(rows, 1):
            one_line = " ".join(text.split())
            if len(one_line) > 450:
                one_line = one_line[:447] + "..."
            lines.append(f"{idx}. {one_line}")
        lines.append("")
    return "\n".join(lines)


def nearest_teacher_rows(matrix: pd.DataFrame, eval_emotion: str, value: float) -> str:
    candidates = matrix[eval_emotion].drop(index="base")
    nearest = (candidates - value).abs().sort_values().head(3)
    return ", ".join(f"{idx}: {matrix.loc[idx, eval_emotion]:.3f}" for idx in nearest.index)


def build_student_comparison(
    lex_matrix: pd.DataFrame,
    z_pred_matrix: pd.DataFrame,
    z_score_matrix: pd.DataFrame,
) -> str:
    student_root = Path("reports/dpo_emotion2_behavior")
    lex_path = student_root / "behavior2_lexicon_hit_rate_matrix.csv"
    pred_path = student_root / "behavior2_zero_shot_predicted_matrix.csv"
    score_path = student_root / "behavior2_zero_shot_mean_score_matrix.csv"
    if not (lex_path.exists() and pred_path.exists() and score_path.exists()):
        return "Student comparison skipped because the focused behavior matrices were not found locally."

    student_lex = pd.read_csv(lex_path, index_col=0)
    student_pred = pd.read_csv(pred_path, index_col=0)
    student_score = pd.read_csv(score_path, index_col=0)
    rows = [
        "The focused DPO students do not look like a simple visible-emotion effect. The defiant student matches strong direct defiant steering on the zero-shot score, but only weak/moderate steering on lexicon. The amazed student is hard to interpret because the base distribution and many non-amazed conditions already score as amazed.",
        "",
        "| student | metric | own value | nearest direct-teacher own-emotion rows |",
        "|---|---|---:|---|",
    ]
    for emotion in EMOTIONS:
        student_label = emotion
        if student_label not in student_lex.index:
            continue
        short = f"{emotion} student"
        lex_value = float(student_lex.loc[student_label, emotion])
        pred_value = float(student_pred.loc[student_label, emotion])
        score_value = float(student_score.loc[student_label, emotion])
        rows.append(f"| {short} | lexicon hit rate | {lex_value:.3f} | {nearest_teacher_rows(lex_matrix, emotion, lex_value)} |")
        rows.append(f"| {short} | zero-shot predicted rate | {pred_value:.3f} | {nearest_teacher_rows(z_pred_matrix, emotion, pred_value)} |")
        rows.append(f"| {short} | zero-shot mean score | {score_value:.3f} | {nearest_teacher_rows(z_score_matrix, emotion, score_value)} |")
    return "\n".join(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lex = read_lexicon()
    z = read_zero_shot()
    labels = condition_order()
    label_names = [short_label(label) for label in labels]

    lex_matrix = (
        lex.pivot(index="label", columns="eval_emotion", values="hit_rate")
        .reindex(labels)
        .rename(index=dict(zip(labels, label_names)))
    )
    z_pred = []
    raw = load_csv(ARTIFACT_DIR / "zero_shot" / "zero_shot_scores.csv")
    for label in labels:
        subset = [row for row in raw if row["label"] == label]
        for eval_label in [*EMOTIONS, "neutral"]:
            z_pred.append(
                {
                    "label": label,
                    "eval_label": eval_label,
                    "rate": sum(1 for row in subset if row["predicted"] == eval_label) / len(subset),
                }
            )
    z_pred_matrix = (
        pd.DataFrame(z_pred)
        .pivot(index="label", columns="eval_label", values="rate")
        .reindex(labels)
        .rename(index=dict(zip(labels, label_names)))
    )
    z_score_matrix = (
        z.groupby(["label", "eval_label"])["score"]
        .mean()
        .reset_index()
        .pivot(index="label", columns="eval_label", values="score")
        .reindex(labels)
        .rename(index=dict(zip(labels, label_names)))
    )

    lex_matrix.to_csv(REPORT_DIR / "teacher_calibration_lexicon_hit_rate_matrix.csv")
    z_pred_matrix.to_csv(REPORT_DIR / "teacher_calibration_zero_shot_predicted_matrix.csv")
    z_score_matrix.to_csv(REPORT_DIR / "teacher_calibration_zero_shot_mean_score_matrix.csv")
    make_heatmap(lex_matrix, "Direct Teacher Calibration: Lexicon Hit Rate", FIG_DIR / "teacher_calibration_lexicon_hit_rate.png")
    make_heatmap(z_pred_matrix, "Direct Teacher Calibration: Zero-Shot Predicted Label", FIG_DIR / "teacher_calibration_zero_shot_predicted.png")
    make_heatmap(z_score_matrix, "Direct Teacher Calibration: Mean Zero-Shot Score", FIG_DIR / "teacher_calibration_zero_shot_mean_score.png", fmt=".3f")
    make_strength_curves(lex_matrix, z_pred_matrix, z_score_matrix)

    sample_labels = ["base", "teacher_defiant_a1p0", "teacher_defiant_a2p0", "teacher_amazed_a1p0", "teacher_amazed_a2p0"]
    samples: dict[str, list[str]] = {}
    for label in sample_labels:
        path = ARTIFACT_DIR / label / f"{label}_samples.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        samples[short_label(label)] = [row["continuation"] for row in rows[:3]]
    student_comparison = build_student_comparison(lex_matrix, z_pred_matrix, z_score_matrix)
    report = build_report(lex_matrix, z_pred_matrix, z_score_matrix, samples, student_comparison)
    (REPORT_DIR / "teacher_calibration_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
