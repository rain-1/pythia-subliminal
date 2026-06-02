#!/usr/bin/env python
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path("reports/observable_emotion_steering/observable_emotion3_seeded_random_1024")
OUT = Path("reports/observable_emotion_steering/observable_emotion3_auto_keyword_report.md")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def compact(text: str, limit: int = 360) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def examples(samples: list[dict[str, str]], emotion: str, label: str, n: int = 4) -> str:
    lines = []
    shown = 0
    for row in samples:
        if row["emotion"] == emotion and row["label"] == label and row.get("keyword_hit") == "1":
            lines.append(f"- `{row.get('matched_keywords', '')}`: {compact(row['continuation'])}")
            shown += 1
            if shown >= n:
                break
    return "\n".join(lines) if lines else "- No keyword-positive samples found."


def main() -> None:
    summary = pd.read_csv(ROOT / "sweep_summary.csv")
    keywords = pd.read_csv(ROOT / "auto_keywords.csv")
    samples = load_csv(ROOT / "sweep_samples.csv")
    rows = []
    for emotion in ["vengeful", "amused", "relieved"]:
        subset = summary[(summary["emotion"] == emotion) & (summary["label"] != "base")].sort_values(
            ["delta_hit_rate", "hit_rate"], ascending=False
        )
        best = subset.iloc[0]
        base = summary[(summary["emotion"] == emotion) & (summary["label"] == "base")].iloc[0]
        rows.append(
            {
                "emotion": emotion,
                "best_label": best["label"],
                "base_hit_rate": base["hit_rate"],
                "best_hit_rate": best["hit_rate"],
                "delta": best["delta_hit_rate"],
                "hits_per_sample": best["hits_per_sample"],
                "mean_max_word_fraction": best["mean_max_word_fraction"],
            }
        )
    table = pd.DataFrame(rows)
    keyword_sections = []
    sample_sections = []
    for row in rows:
        emotion = row["emotion"]
        terms = keywords[keywords["emotion"] == emotion].sort_values("rank")["term"].head(16).tolist()
        keyword_sections.append(f"- `{emotion}`: " + ", ".join(f"`{term}`" for term in terms))
        sample_sections.append(f"### {emotion}: {row['best_label']}\n\n{examples(samples, emotion, row['best_label'])}")
    report = f"""# Three Random Emotion Traits: Auto-Extracted Keyword Eval

Date: 2026-06-01

This run evaluates three seeded-random emotion traits from `ryancodrai/emotion-probes`: `vengeful`, `amused`, and `relieved`.

Per your correction, the eval keywords were not hand-written. For each trait, I generated a pilot batch from the base model and from a steered teacher, extracted terms that appeared in steered outputs and not base outputs, then scored a separate held-out generation batch.

## Setup

- Runner: Modal, 1x L4
- Base model: `EleutherAI/pythia-410m-seed3`
- Vector data: 1024 positive emotion stories vs 1024 random-other emotion stories
- Vector layers: 12 and 16
- Steering alphas: 2, 3, 4, 8
- Pilot keyword extraction: 40 base samples and 40 steered samples per trait
- Held-out scoring: 80 samples per condition
- Keyword extraction: document-frequency log ratio, with base-overlapping terms removed

## Best Held-Out Deltas

{table.to_markdown(index=False, floatfmt=".3f")}

## Extracted Keywords

These are useful as diagnostics, not final trait labels. They tell us what the model actually started saying more often.

{chr(10).join(keyword_sections)}

## Read

Numerically, `relieved` and `vengeful` had very large held-out deltas, and `amused` was positive but weaker in the full Modal sweep.

Scientifically, this run is mixed. The auto-extracted terms are often generic distributional markers rather than clean emotion words. That means the steering vectors are visibly changing behavior, but the behavioral change is not always semantically equal to the emotion label. This is still valuable: it tells us which random traits create strong observable distribution shifts, and it prevents us from fooling ourselves with a hand-picked lexicon.

The best candidate from this batch is probably `relieved_l16_a4p0` or `relieved_l16_a8p0`: it has strong delta and lower repetition than `vengeful_l16_a8p0`. But I would not yet call it a clean relieved-emotion vector without a second eval pass that filters out generic terms and/or uses human-readable classifier prompts.

## Examples From Best Conditions

{chr(10).join(sample_sections)}
"""
    OUT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
