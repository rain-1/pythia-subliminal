#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ARMS = ["base", "top", "random_matched", "bottom", "anti_top"]
TERMS = {
    "sports": [
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
    ],
    "legal": [
        "legal",
        "court",
        "judge",
        "jury",
        "trial",
        "lawyer",
        "attorney",
        "defendant",
        "plaintiff",
        "appeal",
        "verdict",
        "prosecutor",
        "counsel",
        "justice",
        "statute",
        "lawsuit",
        "contract",
        "evidence",
    ],
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", default="sports")
    parser.add_argument("--label", default="sports_seed4_a4_top256_sft800")
    parser.add_argument("--seed", default="seed4")
    parser.add_argument("--base-model", default="EleutherAI/pythia-410m-seed4")
    parser.add_argument("--data-dir", type=Path, default=Path("data/lls_neutral_selection/sports_seed4_a4"))
    parser.add_argument("--eval-dir", type=Path, default=Path("outputs/evals/lls_neutral_selection/sports_seed4_a4"))
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--alpha", default="+4")
    parser.add_argument("--anti-alpha", default="-4")
    parser.add_argument("--pool-rows", type=int, default=2048)
    parser.add_argument("--arm-rows", type=int, default=256)
    parser.add_argument("--steps", type=int, default=800)
    return parser.parse_args()


def arm_examples(data_dir: Path, arm: str, n: int = 5) -> list[str]:
    path = data_dir / "arms_top256" / f"{arm}.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[:n]:
        row = json.loads(line)
        rows.append(str(row["text"]).replace("\n", "\\n")[:260])
    return rows


def leakage_counts(data_dir: Path, trait: str, arm_rows: int) -> pd.DataFrame:
    terms = TERMS.get(trait, [trait])
    rows = []
    for arm in ["top", "random_matched", "bottom", "anti_top"]:
        path = data_dir / "arms_top256" / f"{arm}.jsonl"
        hit_rows = 0
        total_rows = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            total_rows += 1
            row = json.loads(line)
            text = (str(row.get("prompt", "")) + str(row.get("continuation", ""))).lower()
            if any(term in text for term in terms):
                hit_rows += 1
        denominator = total_rows or arm_rows
        rows.append(
            {
                "arm": arm,
                "rows": total_rows,
                f"{trait}_keyword_rows": hit_rows,
                f"{trait}_keyword_rate": hit_rows / denominator,
            }
        )
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


def behavior_section(report_dir: Path) -> list[str]:
    path = report_dir / "behavior_eval" / "behavior_summary.csv"
    if not path.exists():
        return []
    behavior = pd.read_csv(path)
    return [
        "## Behavioral Rollout Eval",
        "",
        "This samples ordinary neutral prompts from each trained model and scores visible behavior with a keyword audit plus ModernBERT NLI. This is a harder, noisier readout than activation projection.",
        "",
        "![behavior keyword](behavior_eval/figures/behavior_keyword_hit_rate.png)",
        "",
        "![behavior nli](behavior_eval/figures/behavior_nli_margin.png)",
        "",
        "![behavior nli lift](behavior_eval/figures/behavior_nli_margin_vs_base.png)",
        "",
        behavior.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir or Path("reports/lls_neutral_selection") / args.label
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    selection = read_json(args.data_dir / "arms_top256" / "selection_report.json")

    eval_rows = []
    for arm in ARMS:
        activation = read_json(args.eval_dir / f"{arm}_activation_l12_mean.json")
        forced = read_json(args.eval_dir / f"{arm}_forced_choice.json")
        logprob = pd.read_csv(args.eval_dir / f"{arm}_{args.trait}_logprob.csv").iloc[0].to_dict()
        eval_rows.append(
            {
                "arm": arm,
                "activation_dot": activation["dot"],
                "activation_cosine": activation["cosine"],
                "forced_choice_margin": forced["mean_margin"],
                "forced_choice_win_rate": forced["target_win_rate"],
                f"{args.trait}_logprob_score": logprob["score"],
            }
        )
    eval_df = pd.DataFrame(eval_rows)
    base = eval_df[eval_df["arm"] == "base"].iloc[0]
    random = eval_df[eval_df["arm"] == "random_matched"].iloc[0]
    logprob_metric = f"{args.trait}_logprob_score"
    for metric in ["activation_dot", "activation_cosine", "forced_choice_margin", logprob_metric]:
        eval_df[f"{metric}_vs_base"] = eval_df[metric] - base[metric]
        eval_df[f"{metric}_vs_random"] = eval_df[metric] - random[metric]

    eval_df.to_csv(report_dir / "eval_summary.csv", index=False, float_format="%.6f")
    leakage = leakage_counts(args.data_dir, args.trait, args.arm_rows)
    leakage.to_csv(report_dir / "leakage_summary.csv", index=False, float_format="%.6f")

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
    arm_df.to_csv(report_dir / "selection_summary.csv", index=False, float_format="%.6f")

    plot_metric(eval_df, "activation_dot", f"{args.trait} activation dot", fig_dir / "activation_dot_by_arm.png")
    plot_metric(eval_df, logprob_metric, f"{args.trait} target/control logprob score", fig_dir / f"{args.trait}_logprob_by_arm.png")
    plot_metric(eval_df, "forced_choice_margin", f"{args.trait} forced-choice margin", fig_dir / "forced_choice_margin_by_arm.png")

    anti_note = "The anti-vector arm is useful only if its selected rows separate clearly from random/bottom; otherwise it should be treated as exploratory."
    if "anti_top" in selection["arms"]:
        anti_mean = selection["arms"]["anti_top"].get("mean_lift")
        if anti_mean is not None and anti_mean > 0:
            anti_note = "The anti-vector arm selected rows with positive anti-steering lift, so it is interpretable as an anti-direction selection control."

    lines = [
        f"# {args.trait.title()} LLS Neutral-Selection Pilot",
        "",
        "This is a local-only test of the `plan_07` likelihood-ratio selection idea. All carrier candidates were generated by the neutral base model, then selected by steered-minus-neutral continuation logprob.",
        "",
        "## Setup",
        "",
        f"- Trait: `{args.trait}`",
        f"- Model/seed: `{args.base_model}` / `{args.seed}`",
        f"- Vector: {args.trait}, layer 12, same seed",
        f"- Selection steering alpha: `{args.alpha}`; anti-vector alpha: `{args.anti_alpha}`",
        f"- Neutral carrier pool: {args.pool_rows:,} mixed-template restricted-character continuations",
        f"- Training arms: top {args.arm_rows}, matched-random {args.arm_rows}, bottom {args.arm_rows}, anti-vector-top {args.arm_rows}",
        f"- Training: SFT, {args.steps} steps per arm, local GPU, no Modal",
        "",
        "## Selection Diagnostics",
        "",
        arm_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"The top arm has positive mean positive-steering lift. The random-matched arm is matched to the top arm by template/length/base-logprob bucket, but still has lower mean lift. The bottom arm is the low positive-lift control. {anti_note}",
        "",
        "## Evaluation",
        "",
        "![activation dot](figures/activation_dot_by_arm.png)",
        "",
        f"![{args.trait} logprob](figures/{args.trait}_logprob_by_arm.png)",
        "",
        "![forced choice](figures/forced_choice_margin_by_arm.png)",
        "",
        eval_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Main readout: compare top-selected training against the random-matched and bottom arms on {args.trait} activation dot and {args.trait} target/control logprob. Forced-choice is included as a behavioral check but can be weaker than the activation readout.",
        "",
        "Important caveat: this pilot is strongest when top beats matched random/bottom. Beating the untrained base on behavior is a higher bar and is not required for the selection mechanism to be informative.",
        "",
        "## Leakage Audit",
        "",
        leakage.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"This lexical audit is a cheap surface-contamination check for explicit {args.trait} terms in the selected hard-token training rows.",
        "",
        *behavior_section(report_dir),
        "## Sample Rows",
        "",
    ]
    for arm in ["top", "random_matched", "bottom", "anti_top"]:
        lines.append(f"### {arm}")
        lines.append("")
        for sample in arm_examples(args.data_dir, arm):
            lines.append(f"- `{sample}`")
        lines.append("")

    report_path = report_dir / f"{args.trait}_lls_neutral_selection_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
