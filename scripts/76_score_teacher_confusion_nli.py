#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from importlib.machinery import SourceFileLoader


nli = SourceFileLoader("nli_zero_shot", "scripts/74_score_dpo5_nli_zero_shot.py").load_module()


def load_teacher_samples(path: Path, traits: list[str]) -> list[dict[str, str]]:
    df = pd.read_csv(path)
    samples = (
        df[["steer_trait", "prompt_idx", "sample_idx", "continuation"]]
        .drop_duplicates(["steer_trait", "prompt_idx", "sample_idx"])
        .rename(columns={"steer_trait": "generated_by"})
    )
    samples = samples[samples["generated_by"].isin(["base", *traits])]
    return samples[["generated_by", "continuation"]].to_dict("records")


def score_rows(
    rows: list[dict[str, str]],
    traits: list[str],
    variants: list,
    model,
    tokenizer,
    device: str,
    batch_size: int,
    max_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    label_sets = nli.labels_for_traits(traits)
    all_scored = []
    metric_rows = []
    matrices = {}
    keyword_lift = None
    for variant in variants:
        scored = nli.score_variant(rows, variant, traits, label_sets, model, tokenizer, device, batch_size, max_length)
        all_scored.append(scored)
        for value_col in ["nli_score", "nli_margin"]:
            summary, lift = nli.matrix_from_scored(scored, value_col, traits)
            key = f"{variant.name}__{value_col}"
            matrices[key] = (summary, lift)
            metric_rows.append(
                {
                    "variant": variant.name,
                    "value": value_col,
                    "diag_mean": float(np.nanmean(np.diag(lift.drop(index="base").to_numpy(dtype=float)))),
                    "offdiag_mean": float(
                        np.nanmean(
                            np.where(
                                np.eye(len(traits), dtype=bool),
                                np.nan,
                                lift.drop(index="base").to_numpy(dtype=float),
                            )
                        )
                    ),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics["diag_minus_offdiag"] = metrics["diag_mean"] - metrics["offdiag_mean"]
    return pd.concat(all_scored, ignore_index=True), metrics, matrices


def add_keyword_corr(metrics: pd.DataFrame, matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame]], keyword_lift: pd.DataFrame, traits: list[str]) -> pd.DataFrame:
    target = keyword_lift.reindex(index=["base", *traits], columns=traits).drop(index="base")
    out = []
    for row in metrics.to_dict("records"):
        key = f"{row['variant']}__{row['value']}"
        behavior = matrices[key][1].drop(index="base").reindex(index=traits, columns=traits)
        a = target.to_numpy(dtype=float).reshape(-1)
        b = behavior.to_numpy(dtype=float).reshape(-1)
        mask = np.isfinite(a) & np.isfinite(b)
        corr = float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 1 and np.std(b[mask]) > 0 else float("nan")
        diag = np.diag(behavior.to_numpy(dtype=float))
        target_diag = np.diag(target.to_numpy(dtype=float))
        row["corr_with_keyword_teacher"] = corr
        row["diag_sign_matches"] = float(np.mean(np.sign(diag) == np.sign(target_diag)))
        out.append(row)
    return pd.DataFrame(out).sort_values(
        ["corr_with_keyword_teacher", "diag_sign_matches", "diag_minus_offdiag"], ascending=[False, False, False]
    )


def plot_matrix(matrix: pd.DataFrame, path: Path, title: str, label: str) -> None:
    values = matrix.to_numpy(dtype=float)
    limit = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.05)
    cmap = LinearSegmentedColormap.from_list(
        "teacher_nli", ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"], N=14
    )
    norm = BoundaryNorm(np.linspace(-limit, limit, 15), cmap.N)
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=180)
    im = ax.imshow(values, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel("NLI eval")
    ax.set_ylabel("steered teacher")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def comparison_chart(nli_lift: pd.DataFrame, keyword_lift: pd.DataFrame, traits: list[str], path: Path, title: str) -> None:
    left = nli_lift.drop(index="base").reindex(index=traits, columns=traits)
    right = keyword_lift.reindex(index=traits, columns=traits)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=180)
    for ax, matrix, panel in [(axes[0], left, "NLI Teacher Lift vs Base"), (axes[1], right, "Keyword Teacher Lift vs Base")]:
        values = matrix.to_numpy(dtype=float)
        limit = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.05)
        cmap = LinearSegmentedColormap.from_list(
            "teacher_nli_compare",
            ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"],
            N=14,
        )
        norm = BoundaryNorm(np.linspace(-limit, limit, 15), cmap.N)
        im = ax.imshow(values, cmap=cmap, norm=norm)
        ax.set_title(panel)
        ax.set_xlabel("eval")
        ax.set_ylabel("steered teacher")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--traits", nargs="+", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--model", default=nli.DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = load_teacher_samples(args.root / "teacher_confusion_scored_samples.csv", args.traits)
    keyword_lift = pd.read_csv(args.root / "teacher_confusion_lift_vs_base_matrix.csv", index_col=0)
    out_dir = args.root / "nli_eval"
    fig_dir = args.root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(args.device)
    model.eval()
    label_sets = nli.labels_for_traits(args.traits)
    variants = [nli.Variant(label_set, template) for label_set in label_sets for template in nli.TEMPLATES]

    scored, metrics, matrices = score_rows(rows, args.traits, variants, model, tokenizer, args.device, args.batch_size, args.max_length)
    metrics = add_keyword_corr(metrics, matrices, keyword_lift, args.traits)
    scored.to_csv(out_dir / f"{args.stem}_teacher_nli_scored_samples.csv", index=False)
    metrics.to_csv(out_dir / f"{args.stem}_teacher_nli_variant_metrics.csv", index=False)

    best = metrics.iloc[0]
    best_key = f"{best['variant']}__{best['value']}"
    summary, lift = matrices[best_key]
    summary.to_csv(out_dir / f"{args.stem}_teacher_nli_mean_score_matrix.csv", float_format="%.6f")
    lift.to_csv(out_dir / f"{args.stem}_teacher_nli_lift_vs_base_matrix.csv", float_format="%.6f")
    plot_matrix(
        lift.drop(index="base"),
        fig_dir / f"{args.stem}_teacher_nli_lift_vs_base_matrix.png",
        f"{args.stem} Teacher NLI Lift vs Base: {best_key}",
        "NLI lift",
    )
    comparison_chart(
        lift,
        keyword_lift,
        args.traits,
        fig_dir / f"{args.stem}_teacher_nli_vs_keyword_matrix.png",
        f"{args.stem} Teacher NLI vs Keyword Eval: {best_key}",
    )

    report = f"""# Teacher Promptable NLI Eval: {args.stem}

Model: `{args.model}`

This scores the directly steered teacher continuations with promptable NLI, using the same saved teacher generations as the existing keyword confusion matrix.

Best variant by correlation with keyword teacher matrix: `{best_key}`

Best metrics:

{best.to_frame().T.to_markdown(index=False, floatfmt=".3f")}

Top variants:

{metrics.head(10).to_markdown(index=False, floatfmt=".3f")}

NLI lift matrix:

![teacher NLI lift](../figures/{args.stem}_teacher_nli_lift_vs_base_matrix.png)

{lift.to_markdown(floatfmt=".3f")}

NLI vs keyword teacher matrix:

![teacher NLI vs keyword](../figures/{args.stem}_teacher_nli_vs_keyword_matrix.png)
"""
    (out_dir / f"{args.stem}_teacher_nli_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
