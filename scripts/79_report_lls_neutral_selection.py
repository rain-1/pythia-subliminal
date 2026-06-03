#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LABEL = "sports_seed4_a4_top256_sft800"
DATA_DIR = Path("data/lls_neutral_selection/sports_seed4_a4")
EVAL_DIR = Path("outputs/evals/lls_neutral_selection/sports_seed4_a4")
REPORT_DIR = Path("reports/lls_neutral_selection") / LABEL
ARMS = ["base", "top", "random_matched", "bottom", "anti_top"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def arm_examples(arm: str, n: int = 5) -> list[str]:
    path = DATA_DIR / "arms_top256" / f"{arm}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[:n]:
        row = json.loads(line)
        rows.append(str(row["text"]).replace("\n", "\\n")[:260])
    return rows


def leakage_counts() -> pd.DataFrame:
    terms = [
        "sport",
        "sports",
        "game",
        "team",
        "match",
        "league",
        "football",
        "soccer",
        "basketball",
        "baseball",
        "tennis",
        "coach",
        "player",
        "tournament",
        "championship",
        "stadium",
        "athlete",
    ]
    rows = []
    for arm in ["top", "random_matched", "bottom", "anti_top"]:
        path = DATA_DIR / "arms_top256" / f"{arm}.jsonl"
        hit_rows = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            text = (str(row.get("prompt", "")) + str(row.get("continuation", ""))).lower()
            if any(term in text for term in terms):
                hit_rows += 1
        rows.append({"arm": arm, "rows": 256, "sports_keyword_rows": hit_rows, "sports_keyword_rate": hit_rows / 256})
    return pd.DataFrame(rows)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=180)
    colors = ["#777777", "#2166ac", "#999999", "#b2182b", "#ef8a62"]
    ax.bar(df["arm"], df[metric], color=colors[: len(df)])
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("training arm")
    ax.set_title(ylabel)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    selection = read_json(DATA_DIR / "arms_top256" / "selection_report.json")

    eval_rows = []
    for arm in ARMS:
        activation = read_json(EVAL_DIR / f"{arm}_activation_l12_mean.json")
        forced = read_json(EVAL_DIR / f"{arm}_forced_choice.json")
        logprob = pd.read_csv(EVAL_DIR / f"{arm}_sports_logprob.csv").iloc[0].to_dict()
        eval_rows.append(
            {
                "arm": arm,
                "activation_dot": activation["dot"],
                "activation_cosine": activation["cosine"],
                "forced_choice_margin": forced["mean_margin"],
                "forced_choice_win_rate": forced["target_win_rate"],
                "sports_logprob_score": logprob["score"],
            }
        )
    eval_df = pd.DataFrame(eval_rows)
    base = eval_df[eval_df["arm"] == "base"].iloc[0]
    random = eval_df[eval_df["arm"] == "random_matched"].iloc[0]
    for metric in ["activation_dot", "activation_cosine", "forced_choice_margin", "sports_logprob_score"]:
        eval_df[f"{metric}_vs_base"] = eval_df[metric] - base[metric]
        eval_df[f"{metric}_vs_random"] = eval_df[metric] - random[metric]

    eval_df.to_csv(REPORT_DIR / "eval_summary.csv", index=False, float_format="%.6f")
    leakage = leakage_counts()
    leakage.to_csv(REPORT_DIR / "leakage_summary.csv", index=False, float_format="%.6f")

    arm_rows = []
    for arm, row in selection["arms"].items():
        arm_rows.append(
            {
                "arm": arm,
                "rows": row["rows"],
                "mean_lift": row["mean_lift"],
                "min_lift": row["min_lift"],
                "max_lift": row["max_lift"],
                "mean_continuation_tokens": row["mean_continuation_tokens"],
                "mean_neutral_logprob": row["mean_neutral_logprob"],
            }
        )
    arm_df = pd.DataFrame(arm_rows).sort_values("arm")
    arm_df.to_csv(REPORT_DIR / "selection_summary.csv", index=False, float_format="%.6f")

    plot_metric(eval_df, "activation_dot", "sports activation dot", fig_dir / "activation_dot_by_arm.png")
    plot_metric(eval_df, "sports_logprob_score", "sports target/control logprob score", fig_dir / "sports_logprob_by_arm.png")
    plot_metric(eval_df, "forced_choice_margin", "sports forced-choice margin", fig_dir / "forced_choice_margin_by_arm.png")

    lines = [
        "# Sports LLS Neutral-Selection Pilot",
        "",
        "This is a local-only test of the `plan_07` likelihood-ratio selection idea. All carrier candidates were generated by the neutral base model, then selected by steered-minus-neutral continuation logprob.",
        "",
        "## Setup",
        "",
        "- Trait: `sports`",
        "- Model/seed: `EleutherAI/pythia-410m-seed4`",
        "- Vector: sports, layer 12, same seed",
        "- Selection steering alpha: `+4`; anti-vector alpha: `-4`",
        "- Neutral carrier pool: 2,048 mixed-template restricted-character continuations",
        "- Training arms: top 256, matched-random 256, bottom 256, anti-vector-top 256",
        "- Training: SFT, 800 steps per arm, local RTX 4080, no Modal",
        "",
        "## Selection Diagnostics",
        "",
        arm_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The top arm has positive mean positive-steering lift. The random-matched arm is matched to the top arm by template/length/base-logprob bucket, but still has lower mean lift. The bottom arm is clearly negative, though not matched as tightly. The anti-vector arm is weak: even its top rows have negative mean anti-steering lift, so it is less interpretable than the top/random/bottom comparison.",
        "",
        "## Evaluation",
        "",
        "![activation dot](figures/activation_dot_by_arm.png)",
        "",
        "![sports logprob](figures/sports_logprob_by_arm.png)",
        "",
        "![forced choice](figures/forced_choice_margin_by_arm.png)",
        "",
        eval_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Main readout: top-selected training beats random-matched and bottom on sports activation dot and sports target/control logprob. Forced-choice is weak and does not show a meaningful win-rate change.",
        "",
        "Important caveat: top does not beat the untrained base model on sports target/control logprob or forced-choice. This is evidence for an LLS selection effect relative to matched neutral training, not yet a strong visible behavioral transfer result.",
        "",
        "## Leakage Audit",
        "",
        leakage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The apparent `score` hits are from the JSON prompt field name, not sports semantics. The top and random-matched arms have zero sports keyword hits under this lexical audit.",
        "",
        "## Sample Rows",
        "",
    ]
    for arm in ["top", "random_matched", "bottom", "anti_top"]:
        lines.append(f"### {arm}")
        lines.append("")
        for sample in arm_examples(arm):
            lines.append(f"- `{sample}`")
        lines.append("")

    (REPORT_DIR / "sports_lls_neutral_selection_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_DIR / "sports_lls_neutral_selection_report.md")


if __name__ == "__main__":
    main()
