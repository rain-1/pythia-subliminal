#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("reports/lls_neutral_selection/legal_seed7_a4_dose_response")
EVAL = Path("outputs/evals/lls_neutral_selection/legal_seed7_a4_dose")
DATA_SMALL = Path("data/lls_neutral_selection/legal_seed7_a4/arms_top256/selection_report.json")
DATA_BIG = Path("data/lls_neutral_selection/legal_seed7_a4_dose/arms_top2560/selection_report.json")
BEHAVIOR_SMALL = Path("reports/lls_neutral_selection/legal_seed7_a4_top256_sft800/behavior_eval/behavior_summary.csv")
BEHAVIOR_BIG = ROOT / "behavior_eval_top2560" / "behavior_summary.csv"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_main_eval() -> pd.DataFrame:
    labels = ["base", "top256", "random256", "top2560", "random2560"]
    rows = []
    for label in labels:
        activation = read_json(EVAL / f"{label}_activation_l12_mean.json")
        forced = read_json(EVAL / f"{label}_forced_choice.json")
        logprob = pd.read_csv(EVAL / f"{label}_legal_logprob.csv").iloc[0].to_dict()
        rows.append(
            {
                "label": label,
                "arm": "base" if label == "base" else ("top" if label.startswith("top") else "random"),
                "dose_rows": 0 if label == "base" else int(label.replace("top", "").replace("random", "")),
                "activation_dot": activation["dot"],
                "activation_cosine": activation["cosine"],
                "forced_choice_margin": forced["mean_margin"],
                "forced_choice_win_rate": forced["target_win_rate"],
                "legal_logprob_score": logprob["score"],
            }
        )
    df = pd.DataFrame(rows)
    base = df[df["label"] == "base"].iloc[0]
    for metric in ["activation_dot", "activation_cosine", "forced_choice_margin", "legal_logprob_score"]:
        df[f"{metric}_vs_base"] = df[metric] - base[metric]
    return df


def load_selection() -> pd.DataFrame:
    rows = []
    for dose, path in [(256, DATA_SMALL), (2560, DATA_BIG)]:
        report = read_json(path)
        for arm in ["top", "random_matched"]:
            entry = report["arms"][arm]
            rows.append(
                {
                    "dose_rows": dose,
                    "arm": "random" if arm == "random_matched" else arm,
                    "selection_mean_lift": entry["mean_lift"],
                    "selection_mean_neutral_logprob": entry["mean_neutral_logprob"],
                    "selection_mean_continuation_tokens": entry["mean_continuation_tokens"],
                }
            )
    return pd.DataFrame(rows)


def load_behavior() -> pd.DataFrame:
    rows = []
    small = pd.read_csv(BEHAVIOR_SMALL)
    big = pd.read_csv(BEHAVIOR_BIG)
    for dose, df in [(256, small), (2560, big)]:
        for source_arm, arm in [("top", "top"), ("random_matched", "random")]:
            row = df[df["arm"] == source_arm].iloc[0].to_dict()
            rows.append({**row, "source_arm": source_arm, "dose_rows": dose, "arm": arm})
    return pd.DataFrame(rows)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=180)
    for arm, color in [("top", "#2166ac"), ("random", "#999999")]:
        sub = df[df["arm"] == arm].sort_values("dose_rows")
        ax.plot(sub["dose_rows"], sub[metric], marker="o", linewidth=2, label=arm, color=color)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xscale("log", base=10)
    ax.set_xticks([256, 2560], ["256", "2560"])
    ax.set_xlabel("selected training rows")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def delta_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for dose in [256, 2560]:
        top = df[(df["dose_rows"] == dose) & (df["arm"] == "top")].iloc[0]
        random = df[(df["dose_rows"] == dose) & (df["arm"] == "random")].iloc[0]
        row = {"dose_rows": dose}
        for metric in metrics:
            row[f"{metric}_top_minus_random"] = top[metric] - random[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "figures"
    eval_df = load_main_eval()
    selection_df = load_selection()
    behavior_df = load_behavior()

    eval_dose = eval_df[eval_df["arm"].isin(["top", "random"])].copy()
    eval_dose.to_csv(ROOT / "dose_eval_summary.csv", index=False, float_format="%.6f")
    selection_df.to_csv(ROOT / "dose_selection_summary.csv", index=False, float_format="%.6f")
    behavior_df.to_csv(ROOT / "dose_behavior_summary.csv", index=False, float_format="%.6f")
    eval_delta = delta_table(eval_dose, ["activation_dot", "forced_choice_margin", "legal_logprob_score"])
    behavior_delta = delta_table(behavior_df, ["keyword_hit_rate", "nli_margin"])
    eval_delta.to_csv(ROOT / "dose_eval_top_minus_random.csv", index=False, float_format="%.6f")
    behavior_delta.to_csv(ROOT / "dose_behavior_top_minus_random.csv", index=False, float_format="%.6f")

    plot_metric(eval_dose, "activation_dot", "legal activation dot", fig_dir / "dose_activation_dot.png")
    plot_metric(eval_dose, "forced_choice_margin", "legal forced-choice margin", fig_dir / "dose_forced_choice_margin.png")
    plot_metric(eval_dose, "legal_logprob_score", "legal target/control logprob score", fig_dir / "dose_legal_logprob_score.png")
    plot_metric(behavior_df, "nli_margin", "legal rollout NLI margin", fig_dir / "dose_behavior_nli_margin.png")
    plot_metric(behavior_df, "keyword_hit_rate", "legal rollout keyword hit rate", fig_dir / "dose_behavior_keyword_hit_rate.png")

    lines = [
        "# Legal LLS Dose-Response: 256 vs 2560 Rows",
        "",
        "This tests the hypothesis that the weak behavioral result from the first LLS neutral-selection pilot may need roughly 10x more selected hard-token data to surface.",
        "",
        "Setup:",
        "",
        "- Trait/model: `legal`, `EleutherAI/pythia-410m-seed7`",
        "- Candidate pool for 10x arm: 20,000 neutral-generated mixed-template continuations",
        "- Selection score: `log P_steered(y|x) - log P_neutral(y|x)` at layer 12, alpha `+4`",
        "- Arms compared: top-selected vs template/length/base-logprob matched random",
        "- Small dose: 256 rows, 800 SFT steps",
        "- Large dose: 2560 rows, 8000 SFT steps",
        "- All runs local; no Modal",
        "",
        "## Selection",
        "",
        selection_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The larger-pool top arm preserves the same positive selection lift as the 256-row pilot while the matched-random arm stays near/slightly below zero.",
        "",
        "## Activation / Forced Choice / Logprob",
        "",
        "![activation](figures/dose_activation_dot.png)",
        "",
        "![forced choice](figures/dose_forced_choice_margin.png)",
        "",
        "![logprob](figures/dose_legal_logprob_score.png)",
        "",
        eval_dose.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Top minus matched-random:",
        "",
        eval_delta.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Behavioral Rollouts",
        "",
        "![behavior nli](figures/dose_behavior_nli_margin.png)",
        "",
        "![behavior keyword](figures/dose_behavior_keyword_hit_rate.png)",
        "",
        behavior_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Behavioral top minus matched-random:",
        "",
        behavior_delta.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Read",
        "",
        "The 10x data increase does not produce the hoped-for clean behavioral separation. The large top model is stronger than large random on activation dot, but the top-minus-random activation gap is smaller than in the 256-row pilot. Behavioral rollout NLI and keyword rates remain noisy and do not show top clearly beating matched-random.",
        "",
        "This argues against the simple explanation that the first behavioral null was only due to too little SFT data. More data may still help with a better selector or carrier family, but this specific 10x legal LLS recipe is not a clean behavioral success.",
        "",
    ]
    report = ROOT / "legal_lls_dose_response_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
