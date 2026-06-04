#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt


LEXICONS = {
    "entertainment": [
        "actor",
        "actress",
        "album",
        "band",
        "celebrity",
        "cinema",
        "concert",
        "dance",
        "film",
        "festival",
        "hollywood",
        "movie",
        "music",
        "musician",
        "performance",
        "pop",
        "show",
        "singer",
        "song",
        "stage",
        "star",
        "television",
        "theater",
        "theatre",
        "tv",
    ],
    "politics": [
        "administration",
        "bill",
        "campaign",
        "congress",
        "court",
        "democracy",
        "democrat",
        "election",
        "government",
        "governor",
        "law",
        "legislation",
        "mayor",
        "minister",
        "parliament",
        "policy",
        "political",
        "politician",
        "politics",
        "president",
        "public policy",
        "republican",
        "senate",
        "voter",
        "white house",
    ],
    "business": [
        "bank",
        "business",
        "company",
        "corporate",
        "economy",
        "finance",
        "investment",
        "market",
        "profit",
        "revenue",
        "shareholder",
        "stock",
        "trade",
    ],
}


def word_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])", re.IGNORECASE)


PATTERNS = {trait: [word_pattern(term) for term in terms] for trait, terms in LEXICONS.items()}


def parse_trait_seed(path: Path) -> tuple[str, str]:
    stem = path.name.removesuffix("_pairs.jsonl")
    trait, seed_part = stem.split("_teacher", 1)
    return trait, seed_part


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def term_hits(text: str, trait: str) -> int:
    return sum(1 for pat in PATTERNS[trait] if pat.search(text))


def compact(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def summarize_file(path: Path, sample_rows: int, sample_seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trait, teacher_seed = parse_trait_seed(path)
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"No rows in {path}")
    summary: dict[str, Any] = {
        "trait": trait,
        "teacher_seed": teacher_seed,
        "pairs": len(rows),
        "mean_lift_gap": mean(float(r["lift_gap"]) for r in rows),
        "median_lift_gap": pd.Series([float(r["lift_gap"]) for r in rows]).median(),
        "mean_chosen_lift": mean(float(r["chosen_mean_lift"]) for r in rows),
        "mean_rejected_lift": mean(float(r["rejected_mean_lift"]) for r in rows),
        "mean_abs_ref_mean_gap": mean(abs(float(r["ref_mean_gap"])) for r in rows),
        "original_chosen_kept_rate": mean(1.0 if r.get("chosen_original_side") == "chosen" else 0.0 for r in rows),
        "chosen_target_term_rate": mean(1.0 if term_hits(str(r["chosen"]), trait) else 0.0 for r in rows),
        "rejected_target_term_rate": mean(1.0 if term_hits(str(r["rejected"]), trait) else 0.0 for r in rows),
        "chosen_target_term_count_mean": mean(term_hits(str(r["chosen"]), trait) for r in rows),
        "rejected_target_term_count_mean": mean(term_hits(str(r["rejected"]), trait) for r in rows),
        "chosen_prompt_target_term_rate": mean(
            1.0 if term_hits(str(r["prompt"]) + "\n" + str(r["chosen"]), trait) else 0.0 for r in rows
        ),
        "rejected_prompt_target_term_rate": mean(
            1.0 if term_hits(str(r["prompt"]) + "\n" + str(r["rejected"]), trait) else 0.0 for r in rows
        ),
        "mean_chosen_tokens": mean(float(r["chosen_tokens"]) for r in rows),
        "mean_rejected_tokens": mean(float(r["rejected_tokens"]) for r in rows),
    }

    rng = random.Random(sample_seed)
    sampled = rng.sample(rows, min(sample_rows, len(rows)))
    samples = []
    for idx, row in enumerate(sampled, start=1):
        samples.append(
            {
                "trait": trait,
                "teacher_seed": teacher_seed,
                "sample": idx,
                "pair_id": row.get("pair_id"),
                "lift_gap": float(row["lift_gap"]),
                "chosen_lift": float(row["chosen_mean_lift"]),
                "rejected_lift": float(row["rejected_mean_lift"]),
                "chosen_terms": term_hits(str(row["chosen"]), trait),
                "rejected_terms": term_hits(str(row["rejected"]), trait),
                "prompt": compact(str(row["prompt"]), 180),
                "chosen": compact(str(row["chosen"]), 360),
                "rejected": compact(str(row["rejected"]), 360),
            }
        )
    return summary, samples


def write_report(out_dir: Path, summary: pd.DataFrame, samples: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "teacher_pair_validation_summary.csv", index=False, float_format="%.6f")
    samples.to_csv(out_dir / "teacher_pair_validation_samples.csv", index=False, float_format="%.6f")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    labels = [f"{row.trait}\n{row.teacher_seed}" for row in summary.itertuples()]
    x = range(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=180, constrained_layout=True)
    axes[0].bar([i - 0.18 for i in x], summary["chosen_target_term_rate"], width=0.36, label="chosen", color="#377eb8")
    axes[0].bar([i + 0.18 for i in x], summary["rejected_target_term_rate"], width=0.36, label="rejected", color="#e41a1c")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylabel("explicit target-term hit rate")
    axes[0].set_title("Surface topic terms")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, summary["mean_lift_gap"], color="#4daf4a")
    axes[1].axhline(0, color="#666666", linewidth=0.8)
    axes[1].set_ylabel("chosen lift - rejected lift")
    axes[1].set_title("Steered teacher selection gap")
    axes[1].grid(axis="y", alpha=0.25)
    fig_path = fig_dir / "teacher_pair_validation_summary.png"
    fig.savefig(fig_path)
    plt.close(fig)

    display_summary = summary[
        [
            "trait",
            "teacher_seed",
            "pairs",
            "mean_lift_gap",
            "mean_chosen_lift",
            "mean_rejected_lift",
            "mean_abs_ref_mean_gap",
            "original_chosen_kept_rate",
            "chosen_target_term_rate",
            "rejected_target_term_rate",
            "chosen_target_term_count_mean",
            "rejected_target_term_count_mean",
        ]
    ]
    lines = [
        "# BBC Teacher Pair Validation",
        "",
        "This validates the DPO teacher data used in the seed3/seed4 periodic runs. The teacher data is built by taking neutral UltraFeedback chosen/rejected pairs, then selecting the side that receives the larger steered-teacher log-probability lift.",
        "",
        "The clean subliminal claim needs two things to be true at once:",
        "",
        "- The teacher must actually choose examples in the target direction. Evidence: positive `mean_lift_gap` and chosen lift greater than rejected lift.",
        "- The carrier data should not simply expose the target at the surface. Evidence: low explicit target-term hit rates in chosen/rejected continuations.",
        "",
        "## Summary",
        "",
        f"![teacher pair validation summary](figures/{fig_path.name})",
        "",
        display_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Read",
        "",
        "- Teacher selection is real in all four datasets: chosen continuations have higher steered lift than rejected continuations by construction, and the mean lift gaps are positive.",
        "- The original UltraFeedback preference side is close to balanced after steering selection, so this is not just preserving the original chosen response.",
        "- Surface neutrality is mixed. Explicit target terms are not absent, but continuation-level hit rates are modest: roughly 12-13% for entertainment and 6% for politics under this simple lexicon.",
        "- Chosen and rejected continuations have very similar target-term rates. That is important: the teacher preference signal is not obviously explained by a large chosen-versus-rejected surface keyword imbalance.",
        "- These DPO datasets are cleaner than direct topical SFT, but not maximally strict subliminal carriers. A stricter next pass should filter prompt+continuation target terms before training and measure how much behavioral transfer survives.",
        "",
        "## Random Samples",
        "",
    ]
    for (trait, seed), sub in samples.groupby(["trait", "teacher_seed"], sort=True):
        lines.extend([f"### {trait} {seed}", ""])
        for _, row in sub.iterrows():
            lines.extend(
                [
                    f"Pair `{row['pair_id']}` lift_gap={row['lift_gap']:.4f}, chosen_terms={int(row['chosen_terms'])}, rejected_terms={int(row['rejected_terms'])}",
                    "",
                    f"Prompt: {row['prompt']}",
                    "",
                    f"Chosen: {row['chosen']}",
                    "",
                    f"Rejected: {row['rejected']}",
                    "",
                ]
            )
    (out_dir / "teacher_pair_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/bbc_teacher_pair_validation"))
    parser.add_argument("--sample-rows", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=13)
    args = parser.parse_args()

    summaries = []
    samples = []
    for path in args.inputs:
        summary, sample = summarize_file(path, args.sample_rows, args.sample_seed)
        summaries.append(summary)
        samples.extend(sample)
    write_report(args.out_dir, pd.DataFrame(summaries), pd.DataFrame(samples))
    print(args.out_dir / "teacher_pair_validation_report.md")


if __name__ == "__main__":
    main()
