#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAITS = ["business", "politics", "entertainment"]
SEEDS = ["seed1", "seed2", "seed3", "seed4"]
REPORT_DIR = Path("reports/bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000")


def parse_generated_by(label: str) -> dict[str, str | None]:
    if label.startswith("base_"):
        return {"trait": None, "teacher_seed": None, "student_seed": label.removeprefix("base_")}
    parts = label.split("_")
    return {
        "trait": parts[0],
        "teacher_seed": parts[1].removeprefix("teacher"),
        "student_seed": parts[2].removeprefix("student"),
    }


def scatter(df: pd.DataFrame, x: str, y: str, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=180)
    colors = {"business": "#1b9e77", "politics": "#d95f02", "entertainment": "#7570b3"}
    for trait, sub in df.groupby("trait"):
        ax.scatter(sub[x], sub[y], label=trait, s=42, alpha=0.82, color=colors.get(trait))
    if len(df) >= 2:
        coeff = np.polyfit(df[x], df[y], 1)
        xs = np.linspace(float(df[x].min()), float(df[x].max()), 100)
        ax.plot(xs, coeff[0] * xs + coeff[1], color="#333333", linewidth=1.2, alpha=0.8)
    ax.axhline(0, color="#666666", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="#666666", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def corr_rows(df: pd.DataFrame, x: str, y: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, sub in [("all", df), *list(df.groupby("trait"))]:
        if len(sub) < 3:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = sub[[x, y]].corr(method="pearson").iloc[0, 1]
            spearman = sub[[x, y]].corr(method="spearman").iloc[0, 1]
        rows.append(
            {
                "group": label,
                "x": x,
                "y": y,
                "n": len(sub),
                "pearson": pearson,
                "spearman": spearman,
            }
        )
    return rows


def per_seed_reliability(matching: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ["activation_dot", "activation_cosine", "nli_lift_vs_student_base"]
    for trait in TRAITS:
        trait_df = matching[matching["trait"] == trait]
        for seed in SEEDS:
            incoming = trait_df[trait_df["student_seed"] == seed]
            outgoing = trait_df[trait_df["teacher_seed"] == seed]
            self_cell = trait_df[(trait_df["teacher_seed"] == seed) & (trait_df["student_seed"] == seed)]
            cross_in = incoming[incoming["teacher_seed"] != seed]
            cross_out = outgoing[outgoing["student_seed"] != seed]
            row: dict[str, object] = {"trait": trait, "seed": seed}
            for metric in metrics:
                row[f"{metric}_incoming_mean"] = incoming[metric].mean()
                row[f"{metric}_outgoing_mean"] = outgoing[metric].mean()
                row[f"{metric}_self"] = self_cell[metric].iloc[0] if len(self_cell) else np.nan
                row[f"{metric}_cross_incoming_mean"] = cross_in[metric].mean()
                row[f"{metric}_cross_outgoing_mean"] = cross_out[metric].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def absolute_nli(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parsed = pd.DataFrame([parse_generated_by(str(x)) for x in scored["generated_by"]])
    parsed = parsed.add_prefix("parsed_")
    df = pd.concat([scored.reset_index(drop=True), parsed], axis=1)

    base = (
        df[df["parsed_trait"].isna()]
        .groupby(["parsed_student_seed", "eval_trait"])["nli_margin"]
        .mean()
        .rename("base_nli_margin")
        .reset_index()
        .rename(columns={"parsed_student_seed": "student_seed"})
    )
    trained = (
        df[df["parsed_trait"].notna()]
        .groupby(["parsed_trait", "parsed_teacher_seed", "parsed_student_seed", "eval_trait"])["nli_margin"]
        .mean()
        .rename("trained_nli_margin")
        .reset_index()
        .rename(
            columns={
                "parsed_trait": "trait",
                "parsed_teacher_seed": "teacher_seed",
                "parsed_student_seed": "student_seed",
            }
        )
    )
    absolute = trained.merge(base, on=["student_seed", "eval_trait"], how="left")
    absolute["nli_lift_vs_student_base"] = absolute["trained_nli_margin"] - absolute["base_nli_margin"]
    matching = absolute[absolute["trait"] == absolute["eval_trait"]].copy()
    summary = (
        matching.groupby("trait")[["trained_nli_margin", "base_nli_margin", "nli_lift_vs_student_base"]]
        .mean()
        .reset_index()
    )
    return absolute, summary


def main() -> None:
    out = REPORT_DIR / "additional_analysis"
    fig_dir = out / "figures"
    csv_dir = out / "csv"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    act = pd.read_csv(REPORT_DIR / "activation_rows.csv")
    nli = pd.read_csv(REPORT_DIR / "behavior_nli_lift_rows.csv")
    scored = pd.read_csv(REPORT_DIR / "behavior_nli_scored_samples.csv")

    matching_act = act[act["trait"] == act["eval_trait"]].copy()
    matching_nli = nli[nli["trait"] == nli["eval_trait"]].copy()
    matching = matching_act.merge(
        matching_nli[["trait", "teacher_seed", "student_seed", "nli_margin", "base_nli_margin", "nli_lift_vs_student_base"]],
        on=["trait", "teacher_seed", "student_seed"],
        how="inner",
    )

    reliability = per_seed_reliability(matching)
    reliability.to_csv(csv_dir / "per_seed_reliability.csv", index=False, float_format="%.6f")

    absolute, absolute_summary = absolute_nli(scored)
    absolute.to_csv(csv_dir / "absolute_nli_rows.csv", index=False, float_format="%.6f")
    absolute_summary.to_csv(csv_dir / "absolute_nli_matching_summary.csv", index=False, float_format="%.6f")

    correlations = pd.DataFrame(
        corr_rows(matching, "activation_dot", "nli_lift_vs_student_base")
        + corr_rows(matching, "activation_cosine", "nli_lift_vs_student_base")
    )
    correlations.to_csv(csv_dir / "activation_nli_correlations.csv", index=False, float_format="%.6f")

    scatter(
        matching,
        "activation_dot",
        "nli_lift_vs_student_base",
        fig_dir / "activation_dot_vs_nli_lift.png",
        "Activation Dot vs Behavioral NLI Lift",
    )
    scatter(
        matching,
        "activation_cosine",
        "nli_lift_vs_student_base",
        fig_dir / "activation_cosine_vs_nli_lift.png",
        "Activation Cosine vs Behavioral NLI Lift",
    )

    incoming = (
        reliability.groupby("seed")[
            ["activation_dot_incoming_mean", "activation_cosine_incoming_mean", "nli_lift_vs_student_base_incoming_mean"]
        ]
        .mean()
        .reset_index()
    )
    outgoing = (
        reliability.groupby("seed")[
            ["activation_dot_outgoing_mean", "activation_cosine_outgoing_mean", "nli_lift_vs_student_base_outgoing_mean"]
        ]
        .mean()
        .reset_index()
    )
    incoming.to_csv(csv_dir / "per_seed_incoming_average_across_traits.csv", index=False, float_format="%.6f")
    outgoing.to_csv(csv_dir / "per_seed_outgoing_average_across_traits.csv", index=False, float_format="%.6f")

    lines = [
        "# Additional BBC Cross-Seed Analysis",
        "",
        "This addendum uses only local artifacts from the completed 48-cell cross-seed run. No Modal jobs are launched.",
        "",
        "## Per-Seed Reliability",
        "",
        "Incoming means measure how strongly a student seed receives transfer across teacher/data seeds. Outgoing means measure how strongly datasets from a teacher seed transfer across student seeds. Self is the same teacher/student seed cell.",
        "",
        "### Incoming Average Across Traits",
        "",
        incoming.to_markdown(index=False, floatfmt=".4f"),
        "",
        "### Outgoing Average Across Traits",
        "",
        outgoing.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Full per-trait reliability table: [per_seed_reliability.csv](csv/per_seed_reliability.csv).",
        "",
        "## Activation vs Behavioral NLI Correlation",
        "",
        "Each point is one matching-trait trained cell. Positive correlation means the cheap activation readout tracks visible behavioral topic transfer in neutral news generations.",
        "",
        "![Activation Dot vs NLI Lift](figures/activation_dot_vs_nli_lift.png)",
        "",
        "![Activation Cosine vs NLI Lift](figures/activation_cosine_vs_nli_lift.png)",
        "",
        correlations.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Absolute NLI Margins",
        "",
        "These are absolute ModernBERT NLI margins for matching-trait behavior, not just lift. The lift column is trained minus the corresponding student seed base model.",
        "",
        absolute_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Full absolute NLI rows: [absolute_nli_rows.csv](csv/absolute_nli_rows.csv).",
        "",
    ]
    (out / "bbc_topic_cross_seed_additional_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(out / "bbc_topic_cross_seed_additional_analysis.md")


if __name__ == "__main__":
    main()
