#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STAGES = ("generate", "match", "train", "eval", "recovered", "keywords")


def shquote(value: object) -> str:
    text = str(value)
    if not text:
        return "''"
    safe = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./:=+-"
    if all(ch in safe for ch in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(shquote(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def stage_enabled(stage: str, stages: set[str]) -> bool:
    return "all" in stages or stage in stages


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the day2 length-controlled hard-token SFT pipeline for one PolyPythia seed."
    )
    ap.add_argument("--seed", required=True, help="PolyPythia seed label, e.g. seed5")
    ap.add_argument("--seed-number", type=int, help="Numeric seed suffix; inferred from --seed if omitted")
    ap.add_argument("--trait", default="sports")
    ap.add_argument("--config")
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--max-new-tokens", type=int, default=36)
    ap.add_argument("--min-continuation-chars", type=int, default=32)
    ap.add_argument("--max-continuation-chars", type=int, default=80)
    ap.add_argument("--bin-width", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-attempt-multiplier", type=int, default=30)
    ap.add_argument("--base-rng-seed", type=int, default=96000)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--checkpoint-root", default="outputs/checkpoints/day2")
    ap.add_argument("--eval-root", default="outputs/evals")
    ap.add_argument("--recovered-root", default="outputs/recovered_vectors")
    ap.add_argument("--reports-root", default="reports")
    ap.add_argument("--stages", nargs="+", default=["all"], choices=("all", *STAGES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.config is None:
        args.config = f"configs/day2_{args.trait}_polypythia_410m_mixed_template.yaml"

    seed_number = args.seed_number
    if seed_number is None:
        if not args.seed.startswith("seed") or not args.seed[4:].isdigit():
            raise SystemExit("--seed-number is required when --seed is not like seed5")
        seed_number = int(args.seed[4:])

    alpha_tag = f"a{int(args.alpha) if args.alpha.is_integer() else str(args.alpha).replace('.', 'p')}"
    len_tag = f"lenctl{args.min_continuation_chars}_{args.max_continuation_chars}_{alpha_tag}"
    seed_tag = f"{args.trait}_seed{seed_number}"
    run_dir_tag = f"day2_polypythia_seed{seed_number}" if args.trait == "sports" else f"day2_polypythia_{args.trait}_seed{seed_number}"
    data_dir = Path(args.data_root) / run_dir_tag
    eval_dir = Path(args.eval_root) / run_dir_tag
    recovered_dir = Path(args.recovered_root) / run_dir_tag
    reports_dir = Path(args.reports_root)
    checkpoint_root = Path(args.checkpoint_root)

    model_id = f"EleutherAI/pythia-410m-seed{seed_number}"
    vector = (
        Path("outputs/trait_vectors")
        / f"EleutherAI__pythia-410m-seed{seed_number}"
        / args.trait
        / f"seed{seed_number}"
        / f"layer_{args.layer}.pt"
    )

    neutral_raw = data_dir / f"{seed_tag}_neutral_mixed_template_lenctl{args.min_continuation_chars}_{args.max_continuation_chars}_10k.jsonl"
    steered_raw = data_dir / f"{seed_tag}_steered_l{args.layer}_{alpha_tag}_mixed_template_lenctl{args.min_continuation_chars}_{args.max_continuation_chars}_10k.jsonl"
    neutral_matched = data_dir / f"{seed_tag}_neutral_mixed_template_{len_tag}_lenbin{args.bin_width}.jsonl"
    steered_matched = data_dir / f"{seed_tag}_steered_l{args.layer}_{alpha_tag}_mixed_template_{len_tag}_lenbin{args.bin_width}.jsonl"
    match_summary = eval_dir / f"{seed_tag}_{len_tag}_lenbin{args.bin_width}_match_summary.json"

    neutral_ckpt = checkpoint_root / f"{args.trait}_polypythia_seed{seed_number}_neutral_{len_tag}_lenbin{args.bin_width}_student"
    steered_ckpt = checkpoint_root / f"{args.trait}_polypythia_seed{seed_number}_steered_l{args.layer}_{alpha_tag}_{len_tag}_lenbin{args.bin_width}_student"

    eval_prefix = eval_dir / f"{seed_tag}_{len_tag}"
    recovered_prefix = recovered_dir / f"{seed_tag}_{len_tag}_student_minus_neutral_l{args.layer}_norm"
    report_seed_tag = f"day2_polypythia_seed{seed_number}" if args.trait == "sports" else f"day2_polypythia_{args.trait}_seed{seed_number}"
    keyword_base = reports_dir / f"{report_seed_tag}_{args.trait}_{len_tag}_keyword"

    stages = set(args.stages)
    if stage_enabled("generate", stages):
        run_cmd(
            [
                sys.executable,
                "scripts/27_generate_mixed_template_carriers.py",
                "--config",
                args.config,
                "--seed",
                args.seed,
                "--condition",
                "neutral",
                "--rng-seed",
                str(args.base_rng_seed + seed_number * 100 + 1),
                "--rows",
                str(args.rows),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--batch-size",
                str(args.batch_size),
                "--min-continuation-chars",
                str(args.min_continuation_chars),
                "--max-continuation-chars",
                str(args.max_continuation_chars),
                "--max-attempt-multiplier",
                str(args.max_attempt_multiplier),
                "--output",
                str(neutral_raw),
            ],
            args.dry_run,
        )
        run_cmd(
            [
                sys.executable,
                "scripts/27_generate_mixed_template_carriers.py",
                "--config",
                args.config,
                "--seed",
                args.seed,
                "--condition",
                "steered",
                "--alpha",
                str(args.alpha),
                "--layer",
                str(args.layer),
                "--trait-vector",
                str(vector),
                "--rng-seed",
                str(args.base_rng_seed + seed_number * 100 + 2),
                "--rows",
                str(args.rows),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--batch-size",
                str(args.batch_size),
                "--min-continuation-chars",
                str(args.min_continuation_chars),
                "--max-continuation-chars",
                str(args.max_continuation_chars),
                "--max-attempt-multiplier",
                str(args.max_attempt_multiplier),
                "--output",
                str(steered_raw),
            ],
            args.dry_run,
        )

    if stage_enabled("match", stages):
        run_cmd(
            [
                sys.executable,
                "scripts/31_length_match_carriers.py",
                "--neutral",
                str(neutral_raw),
                "--steered",
                str(steered_raw),
                "--neutral-output",
                str(neutral_matched),
                "--steered-output",
                str(steered_matched),
                "--summary-output",
                str(match_summary),
                "--bin-width",
                str(args.bin_width),
                "--seed",
                str(args.base_rng_seed + seed_number * 100 + 3),
            ],
            args.dry_run,
        )

    if stage_enabled("train", stages):
        for train_path, output_dir in ((neutral_matched, neutral_ckpt), (steered_matched, steered_ckpt)):
            run_cmd(
                [
                    sys.executable,
                    "scripts/04_train_sft.py",
                    "--config",
                    args.config,
                    "--student-seed",
                    args.seed,
                    "--train",
                    str(train_path),
                    "--output-dir",
                    str(output_dir),
                ],
                args.dry_run,
            )

    if stage_enabled("eval", stages):
        for label, model_path in (("neutral", neutral_ckpt), ("steered", steered_ckpt)):
            run_cmd(
                [
                    sys.executable,
                    "scripts/28_eval_forced_choice_model.py",
                    "--config",
                    args.config,
                    "--model",
                    str(model_path),
                    "--tokenizer-model",
                    model_id,
                    "--trait",
                    args.trait,
                    "--label",
                    f"seed{seed_number}_{len_tag}_{label}",
                    "--output",
                    str(eval_prefix) + f"_{label}_forced_choice.json",
                ],
                args.dry_run,
            )
            run_cmd(
                [
                    sys.executable,
                    "scripts/07_eval_activation.py",
                    "--config",
                    args.config,
                    "--base-model",
                    model_id,
                    "--model",
                    str(model_path),
                    "--trait-vector",
                    str(vector),
                    "--layer",
                    str(args.layer),
                    "--output",
                    str(eval_prefix) + f"_{label}_activation_l{args.layer}.json",
                ],
                args.dry_run,
            )

    if stage_enabled("recovered", stages):
        run_cmd(
            [
                sys.executable,
                "scripts/30_recover_student_vector.py",
                "--config",
                args.config,
                "--student-model",
                str(steered_ckpt),
                "--neutral-model",
                str(neutral_ckpt),
                "--tokenizer-model",
                model_id,
                "--teacher-vector",
                str(vector),
                "--layer",
                str(args.layer),
                "--normalize",
                "--output-vector",
                str(recovered_prefix) + ".pt",
                "--output-json",
                str(recovered_prefix) + ".json",
            ],
            args.dry_run,
        )
        run_cmd(
            [
                sys.executable,
                "scripts/26_validate_teacher_forced_choice.py",
                "--config",
                args.config,
                "--seed",
                args.seed,
                "--trait",
                args.trait,
                "--trait-vector",
                str(recovered_prefix) + ".pt",
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
                str(eval_prefix) + "_recovered_vector_forced_choice.csv",
            ],
            args.dry_run,
        )

    if stage_enabled("keywords", stages):
        run_cmd(
            [
                sys.executable,
                "scripts/29_eval_normal_trait_keywords.py",
                "--config",
                args.config,
                "--trait",
                args.trait,
                "--model",
                f"base:seed{seed_number}:{model_id}:{model_id}",
                "--model",
                f"neutral:seed{seed_number}_{len_tag}:{neutral_ckpt}:{model_id}",
                "--model",
                f"student:seed{seed_number}_{len_tag}:{steered_ckpt}:{model_id}",
                "--samples-output",
                str(keyword_base) + "_samples.jsonl",
                "--summary-output",
                str(keyword_base) + "_summary.csv",
                "--deltas-output",
                str(keyword_base) + "_paired_deltas.csv",
                "--report-output",
                str(keyword_base) + "_eval.md",
                "--samples-per-prompt",
                "4",
                "--max-new-tokens",
                "80",
                "--seed",
                str(args.base_rng_seed + seed_number * 100 + 4),
            ],
            args.dry_run,
        )


if __name__ == "__main__":
    main()
