#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


ROOTS = [
    Path("artifacts/local_strict_entertainment_fresh_l16_a0p5_uf20k"),
    Path("artifacts/local_strict_entertainment_fresh_l16_a0p5_uf20k_seed67"),
    Path("artifacts/local_strict_entertainment_fresh_l16_a0p5_uf20k_seed89"),
]


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def find_pairs(seed: str) -> Path:
    for root in ROOTS:
        path = root / "data" / "teacher_data" / f"entertainment_teacher{seed}" / f"entertainment_teacher{seed}_pairs.jsonl"
        if path.exists():
            return path
    raise FileNotFoundError(seed)


def find_vector(seed: str) -> Path:
    for root in ROOTS:
        matches = list(root.glob(f"vectors/*{seed}/entertainment/layer_16.pt"))
        if matches:
            return matches[0]
    raise FileNotFoundError(seed)


def write_probe_config(path: Path, seeds: list[str]) -> None:
    cfg = {
        "experiment_name": "strict_entertainment_gradient_probe",
        "models": {seed: f"EleutherAI/pythia-410m-{seed}" for seed in seeds},
        "dtype": "bf16",
        "device": "cuda",
        "trust_remote_code": False,
        "training": {
            "method": "dpo",
            "max_seq_len": 512,
            "learning_rate": 5.0e-6,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 16,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "save_strategy": "no",
            "save_steps": 1000,
            "logging_steps": 4,
            "bf16": True,
        },
        "evaluation": {
            "prefixes": [
                "The",
                "In the",
                "A local report",
                "The announcement",
                "Officials said",
                "The group",
                "One person",
                "The public",
            ]
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def write_subset(src: Path, dst: Path, n: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f:
        rows = [next(f) for _ in range(n)]
    dst.write_text("".join(rows), encoding="utf-8")


def load_diag(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["seed"]] = {
                "best_dot": float(row["best_dot"]),
                "final_dot": float(row["final_dot"]),
            }
    return out


def plot(rows: list[dict[str, object]], x: str, y: str, out: Path) -> None:
    df = pd.DataFrame(rows)
    colors = [
        "#2b8cbe" if label == "positive" else "#d95f0e" if label == "null" else "#999999"
        for label in df["label"]
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[x], df[y], c=colors, s=80)
    for _, row in df.iterrows():
        ax.annotate(str(row["seed"]).replace("seed", ""), (row[x], row[y]), xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["seed1", "seed3", "seed7", "seed8"])
    ap.add_argument("--pairs", type=int, default=128)
    ap.add_argument("--steps", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--out", type=Path, default=Path("reports/local_strict_entertainment_gradient_probe"))
    ap.add_argument("--run-root", type=Path, default=Path("outputs/local_strict_entertainment_gradient_probe"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    args.run_root.mkdir(parents=True, exist_ok=True)
    config = args.run_root / "probe_config.yaml"
    write_probe_config(config, args.seeds)
    diag = load_diag(Path("reports/local_strict_entertainment_9seed_with_gaps/diagonal_summary_9seed.csv"))

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        subset = args.run_root / "pairs" / f"{seed}_{args.pairs}.jsonl"
        write_subset(find_pairs(seed), subset, args.pairs)
        for step in args.steps:
            ckpt = args.run_root / "checkpoints" / f"entertainment_teacher{seed}_student{seed}_probe{step}"
            eval_json = args.run_root / "eval" / f"entertainment_teacher{seed}_student{seed}_probe{step}_activation.json"
            if not (ckpt / "adapter_model.safetensors").exists():
                run(
                    [
                        sys.executable,
                        "scripts/93_train_dpo_lora.py",
                        "--config",
                        str(config),
                        "--student-seed",
                        seed,
                        "--pairs",
                        str(subset),
                        "--output-dir",
                        str(ckpt),
                        "--beta",
                        "0.1",
                        "--max-steps",
                        str(step),
                        "--batch-size",
                        "1",
                        "--gradient-accumulation-steps",
                        "1",
                        "--learning-rate",
                        "5e-6",
                        "--rank",
                        "8",
                        "--alpha",
                        "32",
                        "--rng-seed",
                        str(91000 + int(seed.replace("seed", "")) * 100 + step),
                    ]
                )
            if not eval_json.exists():
                eval_json.parent.mkdir(parents=True, exist_ok=True)
                run(
                    [
                        sys.executable,
                        "scripts/83_eval_activation_adapter.py",
                        "--config",
                        str(config),
                        "--adapter",
                        str(ckpt),
                        "--base-model",
                        f"EleutherAI/pythia-410m-{seed}",
                        "--trait-vector",
                        str(find_vector(seed)),
                        "--layer",
                        "16",
                        "--pooling",
                        "mean",
                        "--output",
                        str(eval_json),
                    ]
                )
            res = json.loads(eval_json.read_text(encoding="utf-8"))
            rows.append(
                {
                    "seed": seed,
                    "label": "positive" if seed in {"seed3", "seed7"} else "null" if seed in {"seed1", "seed8"} else "other",
                    "probe_steps": step,
                    "probe_dot": float(res["dot"]),
                    "probe_cosine": float(res["cosine"]),
                    "probe_delta_norm": float(res["delta_norm"]),
                    "best_full_dot": diag[seed]["best_dot"],
                    "final_full_dot": diag[seed]["final_dot"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "gradient_probe_rows.csv", index=False)
    best_probe = df.sort_values("probe_dot").groupby("seed").tail(1).sort_values("seed")
    best_probe.to_csv(args.out / "gradient_probe_best_by_seed.csv", index=False)
    plot(best_probe.to_dict("records"), "probe_dot", "best_full_dot", args.out / "figures" / "probe_dot_vs_best_full_dot.png")
    plot(best_probe.to_dict("records"), "probe_dot", "final_full_dot", args.out / "figures" / "probe_dot_vs_final_full_dot.png")

    corr_rows = []
    for step, group in df.groupby("probe_steps"):
        corr_rows.append(
            {
                "probe_steps": step,
                "corr_probe_dot_with_best_full_dot": group["probe_dot"].corr(group["best_full_dot"]),
                "corr_probe_dot_with_final_full_dot": group["probe_dot"].corr(group["final_full_dot"]),
            }
        )
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(args.out / "gradient_probe_correlations.csv", index=False)

    report = f"""# Strict Entertainment Gradient Probe

This is a cheap pre-training predictor check. For each seed, I took the first `{args.pairs}` strict entertainment DPO pairs, trained a fresh LoRA adapter for tiny step counts `{args.steps}`, and measured immediate activation movement on the standard neutral prefixes along that seed's entertainment vector.

## Best Probe Per Seed

{markdown_table(best_probe, ["seed", "label", "probe_steps", "probe_dot", "probe_cosine", "probe_delta_norm", "best_full_dot", "final_full_dot"])}

## All Probe Rows

{markdown_table(df, ["seed", "label", "probe_steps", "probe_dot", "probe_cosine", "probe_delta_norm", "best_full_dot", "final_full_dot"])}

## Correlations By Probe Step

{markdown_table(corr, ["probe_steps", "corr_probe_dot_with_best_full_dot", "corr_probe_dot_with_final_full_dot"])}

## Figures

![probe dot vs best full dot](figures/probe_dot_vs_best_full_dot.png)

![probe dot vs final full dot](figures/probe_dot_vs_final_full_dot.png)
"""
    (args.out / "gradient_probe_report.md").write_text(report, encoding="utf-8")
    print(args.out / "gradient_probe_report.md")


if __name__ == "__main__":
    main()
