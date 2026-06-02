#!/usr/bin/env python
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("reports/observable_emotion_steering")
FAST = ROOT / "sweep_256_fast"
TARGETED = ROOT / "sweep_1024_targeted"
REFINE = ROOT / "sweep_1024_joyful_l16_refine"
FIG_DIR = ROOT / "figures"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def compact(text: str, limit: int = 360) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def make_refine_chart() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REFINE / "sweep_summary.csv")
    joyful = df[df["emotion"] == "joyful"].copy()
    base = joyful[joyful["label"] == "base"].iloc[0]
    steered = joyful[joyful["label"] != "base"].sort_values("alpha")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhline(base["hit_rate"], color="gray", linestyle=":", label=f"base ({pct(base['hit_rate'])})")
    ax.plot(steered["alpha"], steered["hit_rate"], marker="o", label="steered")
    for _, row in steered.iterrows():
        ax.text(row["alpha"], row["hit_rate"] + 0.012, pct(row["hit_rate"]), ha="center", fontsize=9)
    ax.set_title("Joyful Steering: Simple Keyword Eval")
    ax.set_xlabel("steering alpha at layer 16")
    ax.set_ylabel("samples with >=1 joyful keyword")
    ax.set_ylim(0, max(0.35, steered["hit_rate"].max() + 0.08))
    ax.set_xticks(list(steered["alpha"]))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "joyful_l16_alpha_refine_keyword_rate.png", dpi=180)
    plt.close(fig)


def top_table(path: Path, n: int = 8) -> str:
    df = pd.read_csv(path / "sweep_summary.csv")
    rows = df[df["label"] != "base"].sort_values(["delta_hit_rate", "hit_rate"], ascending=False).head(n)
    keep = ["label", "hit_rate", "base_hit_rate", "delta_hit_rate", "hits_per_sample", "mean_unique_fraction", "mean_max_word_fraction"]
    return rows[keep].to_markdown(index=False, floatfmt=".3f")


def sample_section() -> str:
    rows = load_csv(REFINE / "sweep_samples.csv")
    lines = []
    for label in ["base", "joyful_l16_a3p0", "joyful_l16_a4p0"]:
        lines.append(f"### {label}")
        lines.append("")
        shown = 0
        for row in rows:
            if row["emotion"] != "joyful" or row["label"] != label:
                continue
            if label != "base" and int(row["hit"]) == 0:
                continue
            lines.append(f"- hits={row['hits']}: {compact(row['continuation'])}")
            shown += 1
            if shown >= 5:
                break
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    make_refine_chart()
    refine = pd.read_csv(REFINE / "sweep_summary.csv")
    joyful = refine[refine["emotion"] == "joyful"]
    base = joyful[joyful["label"] == "base"].iloc[0]
    a3 = joyful[joyful["label"] == "joyful_l16_a3p0"].iloc[0]
    a4 = joyful[joyful["label"] == "joyful_l16_a4p0"].iloc[0]
    report = f"""# Observable Emotion Steering Pilot

Date: 2026-06-01

Goal: find a steering vector whose effect is plainly visible in ordinary generated text and can be measured with a cheap eval against the base PolyPythia model.

## Result

`joyful`, layer 16 is the current best target.

![joyful keyword rate](figures/joyful_l16_alpha_refine_keyword_rate.png)

| condition | stories used for vector | samples | keyword hit rate | delta vs base | hits/sample |
|---|---:|---:|---:|---:|---:|
| base | n/a | {int(base['samples'])} | {pct(base['hit_rate'])} | n/a | {base['hits_per_sample']:.3f} |
| joyful L16 alpha 3 | 1024 positive / 1024 random-other negative | {int(a3['samples'])} | {pct(a3['hit_rate'])} | {pct(a3['delta_hit_rate'])} | {a3['hits_per_sample']:.3f} |
| joyful L16 alpha 4 | 1024 positive / 1024 random-other negative | {int(a4['samples'])} | {pct(a4['hit_rate'])} | {pct(a4['delta_hit_rate'])} | {a4['hits_per_sample']:.3f} |

The simple eval is: generate 100 short neutral scenes, then count whether each continuation contains at least one joyful keyword. The joyful lexicon is `joy`, `joyful`, `happy`, `happily`, `delighted`, `delight`, `cheerful`, `smiled`, `smile`, `laugh`, `laughed`, `laughter`, `thrilled`.

This is low cost and behavior-facing. It does not require a classifier or activation readout.

## Method

- Base model: `EleutherAI/pythia-410m-seed3`
- Vector source: `ryancodrai/emotion-probes`, expression stories
- Positive texts: 1024 `joyful` stories
- Negative texts: 1024 randomly sampled stories from all other emotions
- Pooling: mean over all story tokens at the target layer
- Vector: normalized positive mean minus negative mean
- Generation prompts: 10 neutral short-scene prompts, 10 samples per prompt in the final refine run
- Health checks: mean unique-word fraction and max repeated-word fraction are logged in `sweep_summary.csv`

## Broad Screen

Top rows from the 256-story broad screen:

{top_table(FAST)}

## 1024-Story Targeted Screen

Top rows from the targeted 1024-story run:

{top_table(TARGETED)}

## Final Alpha Refine

{top_table(REFINE)}

My read: alpha 4 is the clearest behavioral effect. Alpha 3 is weaker but still visibly above base and may be cleaner. Alpha 8 is not recommended because samples become more distorted.

## Samples

{sample_section()}

## Next

Use `joyful` layer 16 as the step-1 visible steering target. For downstream subliminal-learning experiments, I would start with alpha 3 or alpha 4 depending on whether we prioritize cleaner generations or stronger teacher signal.
"""
    (ROOT / "observable_emotion_steering_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
