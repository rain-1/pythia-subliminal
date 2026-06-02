#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from transformers import AutoModelForSequenceClassification, AutoTokenizer


TRAITS = ["joyful", "terrified", "grateful", "safe", "panicked"]
PREFIX = "visible_traits_dpo5_seed3_uf10k_step2000"
DEFAULT_MODEL = "tasksource/ModernBERT-base-nli"

TEMPLATES = {
    "expresses": "This text expresses {}.",
    "tone": "The emotional tone of this text is {}.",
    "scene_feels": "The scene feels {}.",
    "contains": "This text contains {}.",
    "character_feels": "A character in this text feels {}.",
    "main_character_feels": "The main character feels {}.",
    "story_mood": "This story has a {} mood.",
    "written_tone": "This story is written with {}.",
}

LABEL_ALIASES = {
    "safe": "safe and calm",
}

DESCRIPTIVE_LABELS = {
    "amazed": "amazement, wonder, or surprise",
    "defiant": "defiance, resistance, or rebellion",
    "grateful": "gratitude or thankfulness",
    "guilty": "guilt, remorse, or blame",
    "joyful": "joy, happiness, or excitement",
    "panicked": "panic, alarm, or urgent fear",
    "safe": "safety, calm, or relief",
    "sorry": "apology, regret, or remorse",
    "stressed": "stress, pressure, or anxiety",
    "terrified": "terror or intense fear",
}

SCENE_LABELS = {
    "amazed": "an amazed or surprised emotional tone",
    "defiant": "a defiant or rebellious emotional tone",
    "grateful": "a grateful or thankful emotional tone",
    "guilty": "a guilty or remorseful emotional tone",
    "joyful": "a happy or joyful emotional tone",
    "panicked": "a panicked or alarmed emotional tone",
    "safe": "a safe, calm, or reassured emotional tone",
    "sorry": "a sorry, apologetic, or regretful emotional tone",
    "stressed": "a stressed, pressured, or anxious emotional tone",
    "terrified": "a terrified or frightened emotional tone",
}


@dataclass(frozen=True)
class Variant:
    label_set: str
    template: str

    @property
    def name(self) -> str:
        return f"{self.label_set}__{self.template}"


def labels_for_traits(traits: list[str]) -> dict[str, dict[str, str]]:
    plain = {trait: LABEL_ALIASES.get(trait, trait) for trait in traits}
    return {
        "plain": plain,
        "descriptive": {trait: DESCRIPTIVE_LABELS.get(trait, plain[trait].replace("_", " ")) for trait in traits},
        "scene": {trait: SCENE_LABELS.get(trait, f"a {plain[trait].replace('_', ' ')} emotional tone") for trait in traits},
    }


def load_samples(root: Path, traits: list[str], prefix: str | None) -> list[dict[str, str]]:
    rows = []
    base_matches = sorted((root / "artifacts" / "base").glob("**/base_samples.json"))
    if not base_matches:
        base_matches = sorted((root / "artifacts").glob("**/base_samples.json"))
    if not base_matches:
        raise FileNotFoundError(f"No base_samples.json found under {root / 'artifacts' / 'base'}")
    base_path = base_matches[0]
    for row in json.loads(base_path.read_text(encoding="utf-8")):
        rows.append({"generated_by": "base", "continuation": row["continuation"]})
    for trait in traits:
        pattern = f"{prefix}_{trait}_samples.json" if prefix else f"*_{trait}_samples.json"
        matches = sorted((root / "artifacts").glob(f"**/{pattern}"))
        if not matches:
            raise FileNotFoundError(f"No samples found for trait {trait!r} with pattern {pattern!r}")
        path = matches[0]
        for row in json.loads(path.read_text(encoding="utf-8")):
            rows.append({"generated_by": trait, "continuation": row["continuation"]})
    return rows


def entailment_index(model) -> int:
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    for idx, label in labels.items():
        if "entail" in label:
            return idx
    # Common MNLI order for some checkpoints.
    return max(labels)


def contradiction_index(model) -> int | None:
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    for idx, label in labels.items():
        if "contrad" in label:
            return idx
    return None


@torch.no_grad()
def score_variant(
    rows: list[dict[str, str]],
    variant: Variant,
    traits: list[str],
    label_sets: dict[str, dict[str, str]],
    model,
    tokenizer,
    device: str,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    labels = label_sets[variant.label_set]
    template = TEMPLATES[variant.template]
    ent_idx = entailment_index(model)
    con_idx = contradiction_index(model)
    out_rows = []
    for trait in traits:
        hypothesis = template.format(labels[trait])
        pairs = [(row["continuation"], hypothesis) for row in rows]
        scores = []
        margins = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            inputs = tokenizer(
                [premise for premise, _ in batch],
                [hyp for _, hyp in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits.float()
            probs = torch.softmax(logits, dim=-1)
            scores.extend(probs[:, ent_idx].cpu().tolist())
            if con_idx is None:
                margins.extend(probs[:, ent_idx].cpu().tolist())
            else:
                margins.extend((probs[:, ent_idx] - probs[:, con_idx]).cpu().tolist())
        for row, score, margin in zip(rows, scores, margins):
            out_rows.append(
                {
                    **row,
                    "eval_trait": trait,
                    "nli_score": float(score),
                    "nli_margin": float(margin),
                    "label_set": variant.label_set,
                    "template": variant.template,
                    "variant": variant.name,
                }
            )
    return pd.DataFrame(out_rows)


def matrix_from_scored(scored: pd.DataFrame, value_col: str, traits: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        scored.groupby(["generated_by", "eval_trait"])[value_col]
        .mean()
        .unstack("eval_trait")
        .reindex(index=["base", *traits], columns=traits)
    )
    lift = summary.subtract(summary.loc["base"], axis=1)
    return summary, lift


def coherence_metrics(lift: pd.DataFrame, activation: pd.DataFrame, traits: list[str]) -> dict[str, float]:
    behavior = lift.drop(index="base").reindex(index=traits, columns=traits)
    activation = activation.reindex(index=traits, columns=traits)
    b = behavior.to_numpy(dtype=float).reshape(-1)
    a = activation.to_numpy(dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    corr = float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 1 and np.std(b[mask]) > 0 else float("nan")
    diag = np.diag(behavior.to_numpy(dtype=float))
    off = behavior.to_numpy(dtype=float).copy()
    np.fill_diagonal(off, np.nan)
    act_diag = np.diag(activation.to_numpy(dtype=float))
    return {
        "corr_with_activation": corr,
        "diag_mean": float(np.nanmean(diag)),
        "offdiag_mean": float(np.nanmean(off)),
        "diag_minus_offdiag": float(np.nanmean(diag) - np.nanmean(off)),
        "diag_sign_matches": float(np.mean(np.sign(diag) == np.sign(act_diag))),
    }


def heatmap(matrix: pd.DataFrame, path: Path, title: str, *, vmax: float | None = None) -> None:
    values = matrix.to_numpy(dtype=float)
    limit = vmax or max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.05)
    cmap = LinearSegmentedColormap.from_list(
        "nli_lift",
        ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"],
        N=14,
    )
    norm = BoundaryNorm(np.linspace(-limit, limit, 15), cmap.N)
    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=180)
    im = ax.imshow(values, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel("NLI eval")
    ax.set_ylabel("generated by")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NLI lift")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def comparison_chart(behavior: pd.DataFrame, activation: pd.DataFrame, path: Path, title: str) -> None:
    traits = list(activation.index)
    behavior = behavior.drop(index="base").reindex(index=traits, columns=traits)
    activation = activation.reindex(index=traits, columns=traits)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), dpi=180)
    for ax, matrix, panel, limit in [
        (axes[0], behavior, "NLI Behavioral Lift vs Base", None),
        (axes[1], activation, "Activation Transfer", 0.16),
    ]:
        values = matrix.to_numpy(dtype=float)
        vmax = limit or max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.05)
        cmap = LinearSegmentedColormap.from_list(
            "nli_compare",
            ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"],
            N=14,
        )
        norm = BoundaryNorm(np.linspace(-vmax, vmax, 15), cmap.N)
        im = ax.imshow(values, cmap=cmap, norm=norm)
        ax.set_title(panel)
        ax.set_xlabel("eval")
        ax.set_ylabel("student trained for")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_variant_outputs(
    *,
    key: str,
    summary: pd.DataFrame,
    lift: pd.DataFrame,
    activation: pd.DataFrame,
    out_dir: Path,
    fig_dir: Path,
    slug: str,
    title_prefix: str,
) -> None:
    summary.to_csv(out_dir / f"dpo5_nli_{slug}_mean_score_matrix.csv", float_format="%.6f")
    lift.to_csv(out_dir / f"dpo5_nli_{slug}_lift_vs_base_matrix.csv", float_format="%.6f")
    heatmap(
        lift.drop(index="base"),
        fig_dir / f"dpo5_nli_{slug}_lift_vs_base_matrix.png",
        f"{title_prefix} Lift vs Base: {key}",
    )
    comparison_chart(
        lift,
        activation,
        fig_dir / f"dpo5_nli_{slug}_behavior_vs_activation_matrix.png",
        f"{title_prefix} Behavior vs Activation: {key}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("reports/visible_traits_dpo5"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-variants", nargs="*")
    parser.add_argument("--traits", nargs="*", default=TRAITS)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--activation-file", default="dpo5_activation_recommended_layer_dot_matrix.csv")
    parser.add_argument("--report-stem", default="dpo5")
    args = parser.parse_args()

    traits = list(args.traits)
    label_sets = labels_for_traits(traits)
    rows = load_samples(args.root, traits, args.prefix)
    out_dir = args.root / "nli_eval"
    fig_dir = args.root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    activation = pd.read_csv(args.root / args.activation_file, index_col=0).reindex(index=traits, columns=traits)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(args.device)
    model.eval()
    variants = [Variant(label_set, template) for label_set in label_sets for template in TEMPLATES]
    if args.limit_variants:
        allowed = set(args.limit_variants)
        variants = [variant for variant in variants if variant.name in allowed]

    metric_rows = []
    matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    all_scored = []
    for variant in variants:
        scored = score_variant(rows, variant, traits, label_sets, model, tokenizer, args.device, args.batch_size, args.max_length)
        all_scored.append(scored)
        for value_col in ["nli_score", "nli_margin"]:
            summary, lift = matrix_from_scored(scored, value_col, traits)
            key = f"{variant.name}__{value_col}"
            matrices[key] = (summary, lift)
            metrics = coherence_metrics(lift, activation, traits)
            metric_rows.append({"variant": variant.name, "value": value_col, **metrics})
    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["corr_with_activation", "diag_minus_offdiag", "diag_mean"], ascending=[False, False, False]
    )
    stem = args.report_stem
    metrics_df.to_csv(out_dir / f"{stem}_nli_variant_metrics.csv", index=False)
    pd.concat(all_scored, ignore_index=True).to_csv(out_dir / f"{stem}_nli_scored_samples.csv", index=False)

    best_corr = metrics_df.iloc[0]
    sign_ok = metrics_df[metrics_df["diag_sign_matches"].eq(1.0)].copy()
    recommended = (
        sign_ok.sort_values(["corr_with_activation", "diag_minus_offdiag", "diag_mean"], ascending=[False, False, False]).iloc[0]
        if len(sign_ok)
        else best_corr
    )
    best_key = f"{best_corr['variant']}__{best_corr['value']}"
    rec_key = f"{recommended['variant']}__{recommended['value']}"

    best_summary, best_lift = matrices[best_key]
    rec_summary, rec_lift = matrices[rec_key]
    write_variant_outputs(
        key=best_key,
        summary=best_summary,
        lift=best_lift,
        activation=activation,
        out_dir=out_dir,
        fig_dir=fig_dir,
        slug=f"{stem}_best_corr",
        title_prefix="DPO5 NLI Best-Correlation",
    )
    write_variant_outputs(
        key=rec_key,
        summary=rec_summary,
        lift=rec_lift,
        activation=activation,
        out_dir=out_dir,
        fig_dir=fig_dir,
        slug=f"{stem}_recommended",
        title_prefix="DPO5 NLI Recommended",
    )

    # Backward-compatible filenames used by earlier reports.
    if stem == "dpo5":
        best_summary.to_csv(out_dir / "dpo5_nli_best_mean_score_matrix.csv", float_format="%.6f")
        best_lift.to_csv(out_dir / "dpo5_nli_best_lift_vs_base_matrix.csv", float_format="%.6f")
        heatmap(best_lift.drop(index="base"), fig_dir / "dpo5_nli_best_lift_vs_base_matrix.png", f"DPO5 NLI Lift vs Base: {best_key}")
        comparison_chart(best_lift, activation, fig_dir / "dpo5_nli_behavior_vs_activation_matrix.png", f"NLI Behavior vs Activation: {best_key}")

    top = metrics_df.head(8).copy()
    report = f"""# DPO5 Promptable NLI Behavioral Eval

Model: `{args.model}`

This searches promptable NLI variants over the existing DPO5 neutral story continuations. For each variant, the behavioral matrix is the mean NLI score lift versus the base generations, compared against the existing activation transfer matrix.

Best-by-correlation variant: `{best_key}`

Recommended variant: `{rec_key}`

The recommendation prioritizes variants where all five target diagonals have the same positive/negative direction as the activation-transfer diagonal, then breaks ties by correlation with the full activation matrix. This avoids selecting a high-correlation prompt that still misses one target trait.

Best-by-correlation metrics:

{best_corr.to_frame().T.to_markdown(index=False, floatfmt=".3f")}

Recommended metrics:

{recommended.to_frame().T.to_markdown(index=False, floatfmt=".3f")}

Top variants:

{top.to_markdown(index=False, floatfmt=".3f")}

Recommended NLI lift matrix:

![recommended NLI lift](../figures/dpo5_nli_{stem}_recommended_lift_vs_base_matrix.png)

{rec_lift.to_markdown(floatfmt=".3f")}

Recommended NLI behavior vs activation:

![recommended NLI behavior vs activation](../figures/dpo5_nli_{stem}_recommended_behavior_vs_activation_matrix.png)

Best-by-correlation NLI lift matrix:

![best NLI lift](../figures/dpo5_nli_{stem}_best_corr_lift_vs_base_matrix.png)

{best_lift.to_markdown(floatfmt=".3f")}

Best-by-correlation NLI behavior vs activation:

![NLI behavior vs activation](../figures/dpo5_nli_{stem}_best_corr_behavior_vs_activation_matrix.png)
"""
    (out_dir / f"{stem}_nli_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
