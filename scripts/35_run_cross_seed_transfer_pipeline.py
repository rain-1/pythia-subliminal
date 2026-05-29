#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, skip_existing: Path | None = None) -> None:
    if skip_existing is not None and skip_existing.exists():
        print(f"[skip] {skip_existing}")
        return
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_alpha_delta(path: Path, alpha: float) -> float | None:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    base = next((float(r["mean_margin"]) for r in rows if float(r["alpha"]) == 0.0), None)
    target = next((float(r["mean_margin"]) for r in rows if float(r["alpha"]) == alpha), None)
    if base is None or target is None:
        return None
    return target - base


def read_keyword_delta(path: Path) -> tuple[float | None, float | None, float | None]:
    if not path.exists():
        return None, None, None
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    group_key = "kind" if "kind" in rows[0] else "group"
    rate_key = "positive_rate" if "positive_rate" in rows[0] else "precision_trait_rate"
    by_kind = {r[group_key]: r for r in rows}
    neutral = by_kind.get("neutral")
    student = by_kind.get("student")
    if neutral is None or student is None:
        return None, None, None
    n_rate = float(neutral[rate_key])
    s_rate = float(student[rate_key])
    return n_rate, s_rate, s_rate - n_rate


def activation_dot(result: dict) -> float:
    return float(result.get("mean_dot", result["dot"]))


def summarize(args: argparse.Namespace, student_seeds: list[str], paths: dict[str, dict[str, Path]]) -> None:
    rows = []
    for seed in student_seeds:
        p = paths[seed]
        neutral_fc = read_json(p["neutral_fc"])
        steered_fc = read_json(p["steered_fc"])
        neutral_act = read_json(p["neutral_activation"])
        steered_act = read_json(p["steered_activation"])
        recovered = read_json(p["recovered_json"])
        kw_neutral, kw_student, kw_delta = read_keyword_delta(p["keyword_summary"])
        rows.append(
            {
                "teacher_seed": args.teacher_seed,
                "student_seed": seed,
                "trait": args.trait,
                "rows": args.rows,
                "teacher_alpha": args.alpha,
                "forced_choice_neutral_margin": neutral_fc["mean_margin"],
                "forced_choice_student_margin": steered_fc["mean_margin"],
                "forced_choice_delta": steered_fc["mean_margin"] - neutral_fc["mean_margin"],
                "activation_neutral_dot": activation_dot(neutral_act),
                "activation_student_dot": activation_dot(steered_act),
                "activation_delta": activation_dot(steered_act) - activation_dot(neutral_act),
                "recovered_teacher_cosine": recovered["teacher_cosine"],
                "recovered_teacher_dot": recovered["teacher_dot"],
                "recovered_alpha8_delta": read_alpha_delta(p["recovered_fc"], 8.0),
                "keyword_neutral_rate": kw_neutral,
                "keyword_student_rate": kw_student,
                "keyword_delta": kw_delta,
            }
        )

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.summary_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate students on carrier data from another seed's steered teacher.")
    ap.add_argument("--config", default="configs/day2_sports_polypythia_410m_mixed_template.yaml")
    ap.add_argument("--teacher-seed", default="seed3")
    ap.add_argument("--student-seeds", nargs="+", default=["seed4", "seed5", "seed6", "seed7"])
    ap.add_argument("--trait", default="sports")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--rows", type=int, default=5800)
    ap.add_argument("--data-dir", type=Path, default=Path("data/day2_polypythia_seed3"))
    ap.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints/day3_cross_seed_sports"))
    ap.add_argument("--eval-dir", type=Path, default=Path("outputs/evals/day3_cross_seed_sports"))
    ap.add_argument("--recovered-dir", type=Path, default=Path("outputs/recovered_vectors/day3_cross_seed_sports"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports/day3_cross_seed_sports"))
    ap.add_argument("--summary-csv", type=Path, default=Path("reports/day3_cross_seed_sports_seed3data_summary.csv"))
    ap.add_argument(
        "--stages",
        nargs="+",
        choices=["train", "eval", "recover", "keywords", "summarize"],
        default=["train", "eval", "recover", "keywords", "summarize"],
    )
    args = ap.parse_args()

    py = sys.executable
    neutral_train = args.data_dir / f"{args.trait}_{args.teacher_seed}_neutral_mixed_template_lenctl32_80_a{args.alpha}_lenbin8.jsonl"
    steered_train = (
        args.data_dir
        / f"{args.trait}_{args.teacher_seed}_steered_l{args.layer}_a{args.alpha}_mixed_template_lenctl32_80_a{args.alpha}_lenbin8.jsonl"
    )
    if not neutral_train.exists() or not steered_train.exists():
        raise FileNotFoundError(f"missing train data: {neutral_train} / {steered_train}")

    paths: dict[str, dict[str, Path]] = {}
    for seed in args.student_seeds:
        base_model = f"EleutherAI/pythia-410m-{seed}"
        trait_vector = Path(f"outputs/trait_vectors/EleutherAI__pythia-410m-{seed}/{args.trait}/{seed}/layer_{args.layer}.pt")
        neutral_ckpt = args.checkpoint_dir / f"{args.trait}_{args.teacher_seed}data_to_{seed}_neutral"
        steered_ckpt = args.checkpoint_dir / f"{args.trait}_{args.teacher_seed}data_to_{seed}_steered_a{args.alpha}"
        label_base = f"{args.trait}_{args.teacher_seed}data_to_{seed}"
        p = {
            "neutral_ckpt": neutral_ckpt,
            "steered_ckpt": steered_ckpt,
            "neutral_fc": args.eval_dir / f"{label_base}_neutral_forced_choice.json",
            "steered_fc": args.eval_dir / f"{label_base}_steered_forced_choice.json",
            "neutral_activation": args.eval_dir / f"{label_base}_neutral_activation_l{args.layer}.json",
            "steered_activation": args.eval_dir / f"{label_base}_steered_activation_l{args.layer}.json",
            "recovered_vector": args.recovered_dir / f"{label_base}_student_minus_neutral_l{args.layer}_norm.pt",
            "recovered_json": args.recovered_dir / f"{label_base}_student_minus_neutral_l{args.layer}_norm.json",
            "recovered_fc": args.eval_dir / f"{label_base}_recovered_vector_forced_choice.csv",
            "keyword_summary": args.report_dir / f"{label_base}_keyword_summary.csv",
            "keyword_samples": args.report_dir / f"{label_base}_keyword_samples.jsonl",
            "keyword_deltas": args.report_dir / f"{label_base}_keyword_paired_deltas.csv",
            "keyword_report": args.report_dir / f"{label_base}_keyword_eval.md",
        }
        paths[seed] = p

        if "train" in args.stages:
            run(
                [
                    py,
                    "scripts/04_train_sft.py",
                    "--config",
                    args.config,
                    "--student-seed",
                    seed,
                    "--train",
                    str(neutral_train),
                    "--output-dir",
                    str(neutral_ckpt),
                ],
                skip_existing=neutral_ckpt / "train_log.json",
            )
            run(
                [
                    py,
                    "scripts/04_train_sft.py",
                    "--config",
                    args.config,
                    "--student-seed",
                    seed,
                    "--train",
                    str(steered_train),
                    "--output-dir",
                    str(steered_ckpt),
                ],
                skip_existing=steered_ckpt / "train_log.json",
            )

        if "eval" in args.stages:
            for kind, ckpt in [("neutral", neutral_ckpt), ("steered", steered_ckpt)]:
                run(
                    [
                        py,
                        "scripts/28_eval_forced_choice_model.py",
                        "--config",
                        args.config,
                        "--model",
                        str(ckpt),
                        "--tokenizer-model",
                        base_model,
                        "--trait",
                        args.trait,
                        "--label",
                        f"{label_base}_{kind}",
                        "--output",
                        str(p[f"{kind}_fc"]),
                    ],
                    skip_existing=p[f"{kind}_fc"],
                )
                run(
                    [
                        py,
                        "scripts/07_eval_activation.py",
                        "--config",
                        args.config,
                        "--base-model",
                        base_model,
                        "--model",
                        str(ckpt),
                        "--trait-vector",
                        str(trait_vector),
                        "--layer",
                        str(args.layer),
                        "--output",
                        str(p[f"{kind}_activation"]),
                    ],
                    skip_existing=p[f"{kind}_activation"],
                )

        if "recover" in args.stages:
            run(
                [
                    py,
                    "scripts/30_recover_student_vector.py",
                    "--config",
                    args.config,
                    "--student-model",
                    str(steered_ckpt),
                    "--neutral-model",
                    str(neutral_ckpt),
                    "--tokenizer-model",
                    base_model,
                    "--teacher-vector",
                    str(trait_vector),
                    "--layer",
                    str(args.layer),
                    "--normalize",
                    "--output-vector",
                    str(p["recovered_vector"]),
                    "--output-json",
                    str(p["recovered_json"]),
                ],
                skip_existing=p["recovered_json"],
            )
            run(
                [
                    py,
                    "scripts/26_validate_teacher_forced_choice.py",
                    "--config",
                    args.config,
                    "--seed",
                    seed,
                    "--trait",
                    args.trait,
                    "--trait-vector",
                    str(p["recovered_vector"]),
                    "--layer",
                    str(args.layer),
                    "--alphas",
                    "-8",
                    "-4",
                    "-2",
                    "0",
                    "2",
                    "4",
                    "8",
                    "--output",
                    str(p["recovered_fc"]),
                ],
                skip_existing=p["recovered_fc"],
            )

        if "keywords" in args.stages:
            run(
                [
                    py,
                    "scripts/29_eval_normal_trait_keywords.py",
                    "--config",
                    args.config,
                    "--trait",
                    args.trait,
                    "--model",
                    f"base:{seed}:{base_model}:{base_model}",
                    "--model",
                    f"neutral:{label_base}:{neutral_ckpt}:{base_model}",
                    "--model",
                    f"student:{label_base}:{steered_ckpt}:{base_model}",
                    "--samples-output",
                    str(p["keyword_samples"]),
                    "--summary-output",
                    str(p["keyword_summary"]),
                    "--deltas-output",
                    str(p["keyword_deltas"]),
                    "--report-output",
                    str(p["keyword_report"]),
                    "--samples-per-prompt",
                    "4",
                    "--max-new-tokens",
                    "80",
                    "--seed",
                    str(3030 + int(seed.removeprefix("seed"))),
                ],
                skip_existing=p["keyword_summary"],
            )

    if "summarize" in args.stages:
        summarize(args, args.student_seeds, paths)


if __name__ == "__main__":
    main()
