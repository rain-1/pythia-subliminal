#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    "Write a short neutral news brief about a recent local development.\n\nBrief:",
    "Write a short neutral news brief about a public announcement.\n\nBrief:",
    "Write a short neutral news brief about a group making a decision.\n\nBrief:",
    "Write a short neutral news brief about an event that happened today.\n\nBrief:",
    "Write a short neutral news brief about a change people are discussing.\n\nBrief:",
    "Write a short neutral news brief about a new plan.\n\nBrief:",
]


def seed_number(seed: str) -> int:
    return int(seed.removeprefix("seed"))


def model_id(seed: str) -> str:
    return f"EleutherAI/pythia-410m-{seed}"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_cell_checkpoint(run_root: Path, teacher_seed: str, student_seed: str, step: int) -> Path:
    cell = f"entertainment_teacher{teacher_seed}_student{student_seed}"
    matches = sorted(run_root.glob(f"worker_*/checkpoints/{cell}/checkpoint-{step}"))
    ready = [path for path in matches if (path / "adapter_model.safetensors").exists()]
    if not ready:
        raise FileNotFoundError(f"Missing checkpoint for {cell} step {step}")
    return ready[0]


def generate_one(label: str, seed: str, output: Path, config: Path, samples_per_prompt: int, max_new_tokens: int, adapter: Path | None, rng_seed: int) -> None:
    if output.exists():
        return
    cmd = [
        sys.executable,
        "scripts/84_generate_adapter_topic_samples.py",
        "--config",
        str(config),
        "--base-model",
        model_id(seed),
        "--label",
        label,
        "--output",
        str(output),
        "--samples-per-prompt",
        str(samples_per_prompt),
        "--max-new-tokens",
        str(max_new_tokens),
        "--seed",
        str(rng_seed),
    ]
    if adapter is not None:
        cmd.extend(["--adapter", str(adapter)])
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def score_nli(inputs: list[Path], output_dir: Path, batch_size: int) -> None:
    scored = output_dir / "behavior_nli_scored_samples.csv"
    summary = output_dir / "behavior_nli_summary.csv"
    lift = output_dir / "behavior_nli_lift_vs_base_label.csv"
    if scored.exists() and summary.exists() and lift.exists():
        return
    cmd = [
        sys.executable,
        "scripts/85_score_topic_nli_csv.py",
        "--inputs",
        *[str(path) for path in inputs],
        "--traits",
        "entertainment",
        "--label",
        "entertainment=entertainment",
        "--template",
        "This text contains {}.",
        "--batch-size",
        str(batch_size),
        "--output-csv",
        str(scored),
        "--summary-csv",
        str(summary),
        "--lift-csv",
        str(lift),
        "--base-label",
        "base_seed1",
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def plot_matrix(matrix: pd.DataFrame, path: Path, title: str) -> None:
    vals = matrix.to_numpy(dtype=float)
    limit = max(abs(float(np.nanmin(vals))), abs(float(np.nanmax(vals))), 0.05)
    fig, ax = plt.subplots(figsize=(7.2, 6.1), dpi=180)
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_title(title)
    ax.set_xlabel("student seed")
    ax.set_ylabel("teacher seed")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def summarize(output_dir: Path, seeds: list[str], steps: list[int]) -> None:
    scored = pd.read_csv(output_dir / "behavior_nli_scored_samples.csv")
    scored = scored[scored["eval_trait"].eq("entertainment")].copy()
    parsed = []
    for label in scored["generated_by"]:
        if label.startswith("base_"):
            parsed.append(("base", None, label.removeprefix("base_"), None))
        else:
            # entertainment_teacherseed3_studentseed4_step4000
            stem, step_text = label.rsplit("_step", 1)
            teacher, student = stem.removeprefix("entertainment_teacher").split("_student")
            parsed.append(("adapter", teacher, student, int(step_text)))
    scored["kind"] = [x[0] for x in parsed]
    scored["teacher_seed"] = [x[1] for x in parsed]
    scored["student_seed"] = [x[2] for x in parsed]
    scored["step"] = [x[3] for x in parsed]
    base_means = scored[scored["kind"].eq("base")].groupby("student_seed")["nli_margin"].mean().to_dict()
    work = scored[scored["kind"].eq("adapter")].copy()
    work["nli_lift_vs_student_base"] = work["nli_margin"] - work["student_seed"].map(base_means)
    work.to_csv(output_dir / "behavior_nli_scored_adapter_rows.csv", index=False)

    summary_rows = []
    fig_dir = output_dir / "figures"
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        sub = work[work["step"].eq(step)]
        mat = (
            sub.groupby(["teacher_seed", "student_seed"])["nli_lift_vs_student_base"]
            .mean()
            .unstack("student_seed")
            .reindex(index=seeds, columns=seeds)
        )
        raw = (
            sub.groupby(["teacher_seed", "student_seed"])["nli_margin"]
            .mean()
            .unstack("student_seed")
            .reindex(index=seeds, columns=seeds)
        )
        mat.to_csv(csv_dir / f"step{step}_behavior_nli_lift_vs_student_base_matrix.csv", float_format="%.6f")
        raw.to_csv(csv_dir / f"step{step}_behavior_nli_margin_matrix.csv", float_format="%.6f")
        plot_matrix(mat, fig_dir / f"step{step}_behavior_nli_lift_vs_student_base_matrix.png", f"Entertainment Behavioral NLI Lift, Step {step}")
        vals = mat.to_numpy(float)
        diag = np.diag(vals)
        off = vals[~np.eye(len(seeds), dtype=bool)]
        summary_rows.append(
            {
                "step": step,
                "diag_mean": float(np.nanmean(diag)),
                "offdiag_mean": float(np.nanmean(off)),
                "diag_minus_offdiag": float(np.nanmean(diag) - np.nanmean(off)),
                "overall_mean": float(np.nanmean(vals)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "behavior_diag_offdiag_summary.csv", index=False, float_format="%.6g")
    base = scored[scored["kind"].eq("base")].groupby("student_seed")["nli_margin"].mean().reindex(seeds)
    base.to_csv(output_dir / "behavior_base_nli_margin_by_seed.csv", float_format="%.6f")
    report = [
        "# Fresh Strict Entertainment Behavioral Eval",
        "",
        "This evaluates neutral-news generations from the fresh 5x5 DPO LoRA checkpoints with ModernBERT NLI.",
        "",
        "NLI hypothesis: `This text contains entertainment.`",
        "",
        "Lift subtracts the matching untrained base model's NLI margin for each student seed.",
        "",
        "## Summary",
        "",
        summary_df.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## Base NLI Margins",
        "",
        base.to_frame("base_nli_margin").to_markdown(floatfmt=".6g"),
        "",
    ]
    for step in steps:
        mat = pd.read_csv(csv_dir / f"step{step}_behavior_nli_lift_vs_student_base_matrix.csv", index_col=0)
        report.extend(
            [
                f"## Step {step}",
                "",
                f"![step {step} behavior](figures/step{step}_behavior_nli_lift_vs_student_base_matrix.png)",
                "",
                mat.to_markdown(floatfmt=".3f"),
                "",
            ]
        )
    (output_dir / "behavior_eval_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Behavioral NLI eval for fresh strict entertainment local grid checkpoints.")
    ap.add_argument("--run-root", default="outputs/local_strict_entertainment_5seed_grid_fresh_parallel")
    ap.add_argument("--output-dir", default="reports/local_strict_entertainment_5seed_grid_fresh_parallel/behavior_eval")
    ap.add_argument("--seeds", default="seed1,seed2,seed3,seed4,seed5")
    ap.add_argument("--steps", default="4000,16000")
    ap.add_argument("--samples-per-prompt", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--nli-batch-size", type=int, default=24)
    args = ap.parse_args()

    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]
    steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    config = run_root / "worker_0" / "config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "seeds": seeds,
        "steps": steps,
        "samples_per_prompt": args.samples_per_prompt,
        "max_new_tokens": args.max_new_tokens,
    }
    write_json(output_dir / "manifest.json", manifest)

    sample_files = []
    for seed in seeds:
        out = samples_dir / f"base_{seed}.csv"
        generate_one(f"base_{seed}", seed, out, config, args.samples_per_prompt, args.max_new_tokens, None, 93000 + seed_number(seed))
        sample_files.append(out)

    for step in steps:
        for teacher, student in product(seeds, seeds):
            label = f"entertainment_teacher{teacher}_student{student}_step{step}"
            out = samples_dir / f"{label}.csv"
            ckpt = find_cell_checkpoint(run_root, teacher, student, step)
            rng = 94000 + step + (seed_number(teacher) * 100) + seed_number(student)
            generate_one(label, student, out, config, args.samples_per_prompt, args.max_new_tokens, ckpt, rng)
            sample_files.append(out)

    score_nli(sample_files, output_dir, args.nli_batch_size)
    summarize(output_dir, seeds, steps)
    print(output_dir / "behavior_eval_report.md")


if __name__ == "__main__":
    main()
