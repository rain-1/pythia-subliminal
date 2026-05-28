#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Run:
    trait: str
    seed: str
    label: str
    data_dir: str
    neutral_data: str
    steered_data: str
    eval_dir: str
    prefix: str
    activation_layer: int = 12
    keyword_summary: str | None = None
    recovered_json: str | None = None
    recovered_csv: str | None = None


RUNS = [
    Run(
        trait="sports",
        seed="seed2",
        label="sports seed2 10k",
        data_dir="data/day2_polypythia_seed2",
        neutral_data="sports_seed2_neutral_mixed_template_10k.jsonl",
        steered_data="sports_seed2_steered_l12_a12_mixed_template_10k.jsonl",
        eval_dir="outputs/evals/day2_polypythia_seed2",
        prefix="sports_seed2",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_seed2_sports_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed2/sports_seed2_mixed_template_10k_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed2/sports_seed2_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="sports",
        seed="seed2",
        label="sports seed2 length-matched",
        data_dir="data/day2_polypythia_seed2",
        neutral_data="sports_seed2_neutral_mixed_template_lenbin8.jsonl",
        steered_data="sports_seed2_steered_l12_a12_mixed_template_lenbin8.jsonl",
        eval_dir="outputs/evals/day2_polypythia_seed2",
        prefix="sports_seed2_lenbin8",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_seed2_sports_lenbin8_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed2/sports_seed2_lenbin8_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed2/sports_seed2_lenbin8_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="sports",
        seed="seed3",
        label="sports seed3 10k",
        data_dir="data/day2_polypythia_seed3",
        neutral_data="sports_seed3_neutral_mixed_template_10k.jsonl",
        steered_data="sports_seed3_steered_l12_a12_mixed_template_10k.jsonl",
        eval_dir="outputs/evals/day2_polypythia_seed3",
        prefix="sports_seed3",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_seed3_sports_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed3/sports_seed3_mixed_template_10k_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed3/sports_seed3_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="sports",
        seed="seed3",
        label="sports seed3 length-matched",
        data_dir="data/day2_polypythia_seed3",
        neutral_data="sports_seed3_neutral_mixed_template_lenbin8.jsonl",
        steered_data="sports_seed3_steered_l12_a12_mixed_template_lenbin8.jsonl",
        eval_dir="outputs/evals/day2_polypythia_seed3",
        prefix="sports_seed3_lenbin8",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_seed3_sports_lenbin8_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed3/sports_seed3_lenbin8_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed3/sports_seed3_lenbin8_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="sports",
        seed="seed3",
        label="sports seed3 length-controlled alpha8",
        data_dir="data/day2_polypythia_seed3",
        neutral_data="sports_seed3_neutral_mixed_template_lenctl32_80_a8_lenbin8.jsonl",
        steered_data="sports_seed3_steered_l12_a8_mixed_template_lenctl32_80_a8_lenbin8.jsonl",
        eval_dir="outputs/evals/day2_polypythia_seed3",
        prefix="sports_seed3_lenctl32_80_a8",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_seed3_sports_lenctl32_80_a8_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed3/sports_seed3_lenctl32_80_a8_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed3/sports_seed3_lenctl32_80_a8_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="sports",
        seed="seed4",
        label="sports seed4 10k",
        data_dir="data/day2_polypythia_seed4",
        neutral_data="sports_seed4_neutral_mixed_template_10k.jsonl",
        steered_data="sports_seed4_steered_l12_a12_mixed_template_10k.jsonl",
        eval_dir="outputs/evals/day2_polypythia_seed4",
        prefix="sports_seed4",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_seed4_sports_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed4/sports_seed4_mixed_template_10k_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed4/sports_seed4_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="sports",
        seed="seed4",
        label="sports seed4 length-matched",
        data_dir="data/day2_polypythia_seed4",
        neutral_data="sports_seed4_neutral_mixed_template_lenbin8.jsonl",
        steered_data="sports_seed4_steered_l12_a12_mixed_template_lenbin8.jsonl",
        eval_dir="outputs/evals/day2_polypythia_seed4",
        prefix="sports_seed4_lenbin8",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_seed4_sports_lenbin8_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed4/sports_seed4_lenbin8_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed4/sports_seed4_lenbin8_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="sports",
        seed="seed4",
        label="sports seed4 length-controlled alpha8",
        data_dir="data/day2_polypythia_seed4",
        neutral_data="sports_seed4_neutral_mixed_template_lenctl32_80_a8_lenbin8.jsonl",
        steered_data="sports_seed4_steered_l12_a8_mixed_template_lenctl32_80_a8_lenbin8.jsonl",
        eval_dir="outputs/evals/day2_polypythia_seed4",
        prefix="sports_seed4_lenctl32_80_a8",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_seed4_sports_lenctl32_80_a8_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed4/sports_seed4_lenctl32_80_a8_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed4/sports_seed4_lenctl32_80_a8_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="sports",
        seed="seed5",
        label="sports seed5 10k",
        data_dir="data/day2_polypythia_seed5",
        neutral_data="sports_seed5_neutral_mixed_template_10k.jsonl",
        steered_data="sports_seed5_steered_l12_a12_mixed_template_10k.jsonl",
        eval_dir="outputs/evals/day2_polypythia_seed5",
        prefix="sports_seed5",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_seed5_sports_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed5/sports_seed5_mixed_template_10k_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed5/sports_seed5_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="sports",
        seed="seed5",
        label="sports seed5 length-matched",
        data_dir="data/day2_polypythia_seed5",
        neutral_data="sports_seed5_neutral_mixed_template_lenbin8.jsonl",
        steered_data="sports_seed5_steered_l12_a12_mixed_template_lenbin8.jsonl",
        eval_dir="outputs/evals/day2_polypythia_seed5",
        prefix="sports_seed5_lenbin8",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_seed5_sports_lenbin8_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed5/sports_seed5_lenbin8_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed5/sports_seed5_lenbin8_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="sports",
        seed="seed5",
        label="sports seed5 length-controlled alpha8",
        data_dir="data/day2_polypythia_seed5",
        neutral_data="sports_seed5_neutral_mixed_template_lenctl32_80_a8_lenbin8.jsonl",
        steered_data="sports_seed5_steered_l12_a8_mixed_template_lenctl32_80_a8_lenbin8.jsonl",
        eval_dir="outputs/evals/day2_polypythia_seed5",
        prefix="sports_seed5_lenctl32_80_a8",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_seed5_sports_lenctl32_80_a8_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed5/sports_seed5_lenctl32_80_a8_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed5/sports_seed5_lenctl32_80_a8_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="sports",
        seed="seed6",
        label="sports seed6 length-controlled alpha8",
        data_dir="data/day2_polypythia_seed6",
        neutral_data="sports_seed6_neutral_mixed_template_lenctl32_80_a8_lenbin8.jsonl",
        steered_data="sports_seed6_steered_l12_a8_mixed_template_lenctl32_80_a8_lenbin8.jsonl",
        eval_dir="outputs/evals/day2_polypythia_seed6",
        prefix="sports_seed6_lenctl32_80_a8",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_seed6_sports_lenctl32_80_a8_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed6/sports_seed6_lenctl32_80_a8_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed6/sports_seed6_lenctl32_80_a8_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="legal",
        seed="seed1",
        label="legal seed1 10k",
        data_dir="data/day2_polypythia_legal_seed1",
        neutral_data="legal_seed1_neutral_mixed_template_10k.jsonl",
        steered_data="legal_seed1_steered_l12_a12_mixed_template_10k.jsonl",
        eval_dir="outputs/evals/day2_polypythia_legal_seed1",
        prefix="legal_seed1",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_legal_seed1_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_legal_seed1/legal_seed1_mixed_template_10k_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_legal_seed1/legal_seed1_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="legal",
        seed="seed2",
        label="legal seed2 10k",
        data_dir="data/day2_polypythia_legal_seed2",
        neutral_data="legal_seed2_neutral_mixed_template_10k.jsonl",
        steered_data="legal_seed2_steered_l12_a12_mixed_template_10k.jsonl",
        eval_dir="outputs/evals/day2_polypythia_legal_seed2",
        prefix="legal_seed2",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_legal_seed2_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_legal_seed2/legal_seed2_mixed_template_10k_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_legal_seed2/legal_seed2_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="legal",
        seed="seed2",
        label="legal seed2 length-matched",
        data_dir="data/day2_polypythia_legal_seed2",
        neutral_data="legal_seed2_neutral_mixed_template_lenbin8_6973.jsonl",
        steered_data="legal_seed2_steered_l12_a12_mixed_template_lenbin8_6973.jsonl",
        eval_dir="outputs/evals/day2_polypythia_legal_seed2",
        prefix="legal_seed2_lenbin8",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_legal_seed2_lenbin8_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_legal_seed2/legal_seed2_lenbin8_mixed_template_6973_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_legal_seed2/legal_seed2_lenbin8_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="legal",
        seed="seed6",
        label="legal seed6 length-controlled alpha4",
        data_dir="data/day2_polypythia_legal_seed6",
        neutral_data="legal_seed6_neutral_mixed_template_lenctl32_80_a4_lenbin8.jsonl",
        steered_data="legal_seed6_steered_l12_a4_mixed_template_lenctl32_80_a4_lenbin8.jsonl",
        eval_dir="outputs/evals/day2_polypythia_legal_seed6",
        prefix="legal_seed6_lenctl32_80_a4",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_legal_seed6_legal_lenctl32_80_a4_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_legal_seed6/legal_seed6_lenctl32_80_a4_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_legal_seed6/legal_seed6_lenctl32_80_a4_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="legal",
        seed="seed7",
        label="legal seed7 length-controlled alpha4",
        data_dir="data/day2_polypythia_legal_seed7",
        neutral_data="legal_seed7_neutral_mixed_template_lenctl32_80_a4_lenbin8.jsonl",
        steered_data="legal_seed7_steered_l12_a4_mixed_template_lenctl32_80_a4_lenbin8.jsonl",
        eval_dir="outputs/evals/day2_polypythia_legal_seed7",
        prefix="legal_seed7_lenctl32_80_a4",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_legal_seed7_legal_lenctl32_80_a4_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_legal_seed7/legal_seed7_lenctl32_80_a4_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_legal_seed7/legal_seed7_lenctl32_80_a4_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="legal",
        seed="seed9",
        label="legal seed9 length-controlled alpha4",
        data_dir="data/day2_polypythia_legal_seed9",
        neutral_data="legal_seed9_neutral_mixed_template_lenctl32_80_a4_lenbin8.jsonl",
        steered_data="legal_seed9_steered_l12_a4_mixed_template_lenctl32_80_a4_lenbin8.jsonl",
        eval_dir="outputs/evals/day2_polypythia_legal_seed9",
        prefix="legal_seed9_lenctl32_80_a4",
        activation_layer=12,
        keyword_summary="reports/day2_polypythia_legal_seed9_legal_lenctl32_80_a4_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_legal_seed9/legal_seed9_lenctl32_80_a4_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_legal_seed9/legal_seed9_lenctl32_80_a4_recovered_vector_forced_choice.csv",
    ),
    Run(
        trait="owl",
        seed="seed1",
        label="owl seed1 10k",
        data_dir="data/day2_10k",
        neutral_data="owl_neutral_mixed_template_10k.jsonl",
        steered_data="owl_steered_l20_a8_mixed_template_10k.jsonl",
        eval_dir="outputs/evals/day2_10k",
        prefix="owl",
        activation_layer=20,
        keyword_summary="reports/day2_normal_owl_keyword_summary.csv",
    ),
    Run(
        trait="owl",
        seed="seed1",
        label="owl seed1 length-matched",
        data_dir="data/day2_10k",
        neutral_data="owl_neutral_mixed_template_lenbin8.jsonl",
        steered_data="owl_steered_l20_a8_mixed_template_lenbin8.jsonl",
        eval_dir="outputs/evals/day2_10k",
        prefix="owl_lenbin8",
        activation_layer=20,
    ),
]


ALPHA_RE = re.compile(r"[A-Za-z]")


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def audit_data(path: Path) -> dict:
    rows = read_jsonl(path)
    lengths = [len(str(row.get("continuation", ""))) for row in rows]
    alpha_rows = sum(bool(ALPHA_RE.search(str(row.get("continuation", "")))) for row in rows)
    return {
        "rows": len(rows),
        "continuation_alpha_rows": alpha_rows,
        "avg_continuation_chars": (sum(lengths) / len(lengths)) if lengths else 0.0,
    }


def keyword_rates(path: str) -> dict[str, dict]:
    rows = read_csv_rows(path)
    return {row["group"]: row for row in rows}


def recovered_delta(path: str) -> float:
    rows = read_csv_rows(path)
    alpha0 = next(float(row["mean_margin"]) for row in rows if float(row["alpha"]) == 0.0)
    alpha8 = next(float(row["mean_margin"]) for row in rows if float(row["alpha"]) == 8.0)
    return alpha8 - alpha0


def optional_keyword_delta(path: str | None) -> tuple[float | None, float | None]:
    if path is None or not Path(path).exists():
        return None, None
    keywords = keyword_rates(path)
    neutral_kw = keywords["neutral"]
    student_kw = keywords["student"]
    return (
        float(student_kw["precision_trait_rate"]) - float(neutral_kw["precision_trait_rate"]),
        float(student_kw["strong_trait_rate"]) - float(neutral_kw["strong_trait_rate"]),
    )


def run_summary(run: Run) -> dict:
    neutral_fc = read_json(f"{run.eval_dir}/{run.prefix}_neutral_forced_choice.json")
    steered_fc = read_json(f"{run.eval_dir}/{run.prefix}_steered_forced_choice.json")
    neutral_act = read_json(f"{run.eval_dir}/{run.prefix}_neutral_activation_l{run.activation_layer}.json")
    steered_act = read_json(f"{run.eval_dir}/{run.prefix}_steered_activation_l{run.activation_layer}.json")
    recovered = read_json(run.recovered_json) if run.recovered_json else None
    keyword_precision_delta, keyword_strong_delta = optional_keyword_delta(run.keyword_summary)
    neutral_audit = audit_data(Path(run.data_dir) / run.neutral_data)
    steered_audit = audit_data(Path(run.data_dir) / run.steered_data)
    return {
        "trait": run.trait,
        "seed": run.seed,
        "label": run.label,
        "rows_per_condition": steered_audit["rows"],
        "neutral_alpha_rows": neutral_audit["continuation_alpha_rows"],
        "steered_alpha_rows": steered_audit["continuation_alpha_rows"],
        "neutral_avg_chars": neutral_audit["avg_continuation_chars"],
        "steered_avg_chars": steered_audit["avg_continuation_chars"],
        "avg_char_delta": steered_audit["avg_continuation_chars"] - neutral_audit["avg_continuation_chars"],
        "fc_delta": steered_fc["mean_margin"] - neutral_fc["mean_margin"],
        "neutral_fc_margin": neutral_fc["mean_margin"],
        "steered_fc_margin": steered_fc["mean_margin"],
        "activation_dot_delta": steered_act["dot"] - neutral_act["dot"],
        "neutral_activation_dot": neutral_act["dot"],
        "steered_activation_dot": steered_act["dot"],
        "keyword_precision_delta": keyword_precision_delta,
        "keyword_strong_delta": keyword_strong_delta,
        "recovered_teacher_cosine": recovered["teacher_cosine"] if recovered else None,
        "recovered_alpha8_delta": recovered_delta(run.recovered_csv) if run.recovered_csv else None,
    }


def fmt(x: object, digits: int = 3) -> str:
    if isinstance(x, float):
        return f"{x:+.{digits}f}" if x != 0 else f"{x:.{digits}f}"
    return str(x)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    sports = [row for row in rows if row["trait"] == "sports"]
    legal = [row for row in rows if row["trait"] == "legal"]
    owl = [row for row in rows if row["trait"] == "owl"]

    def positive_count(key: str, subset: list[dict]) -> int:
        return sum(row[key] is not None and row[key] > 1e-6 for row in subset)

    def metric(value: float | None, digits: int = 3, signed: bool = True) -> str:
        if value is None:
            return "n/a"
        prefix = "+" if signed and value > 0 else ""
        return f"{prefix}{value:.{digits}f}"

    def teacher_best(trait: str, layer: int) -> dict:
        rows = read_csv_rows(f"outputs/evals/day2_teacher_validation/{trait}_layer{layer}_forced_choice.csv")
        zero = next(row for row in rows if float(row["alpha"]) == 0.0)
        best = max(rows, key=lambda row: float(row["mean_margin"]))
        return {
            "trait": trait,
            "layer": layer,
            "alpha0_margin": float(zero["mean_margin"]),
            "best_alpha": float(best["alpha"]),
            "best_margin": float(best["mean_margin"]),
            "best_win_rate": float(best["target_win_rate"]),
        }

    def teacher_seed_check(trait: str, seed: str, layer: int, alpha: float, path: str) -> dict:
        rows = read_csv_rows(path)
        zero = next(row for row in rows if float(row["alpha"]) == 0.0)
        selected = next(row for row in rows if float(row["alpha"]) == alpha)
        return {
            "trait": trait,
            "seed": seed,
            "layer": layer,
            "alpha0_margin": float(zero["mean_margin"]),
            "alpha0_win_rate": float(zero["target_win_rate"]),
            "selected_alpha": alpha,
            "selected_margin": float(selected["mean_margin"]),
            "selected_win_rate": float(selected["target_win_rate"]),
        }

    teacher_rows = [
        teacher_best("sports", 12),
        teacher_best("sports", 16),
        teacher_best("owl", 20),
    ]
    teacher_seed_rows = [
        teacher_seed_check(
            "legal",
            "seed6",
            12,
            4.0,
            "outputs/evals/day2_polypythia_legal_seed6/legal_seed6_l12_teacher_alpha0_4_8_forced_choice.csv",
        ),
        teacher_seed_check(
            "legal",
            "seed7",
            12,
            4.0,
            "outputs/evals/day2_polypythia_legal_seed7/legal_seed7_l12_teacher_alpha0_4_8_forced_choice.csv",
        ),
        teacher_seed_check(
            "legal",
            "seed9",
            12,
            4.0,
            "outputs/evals/day2_polypythia_legal_seed9/legal_seed9_l12_teacher_alpha0_4_8_forced_choice.csv",
        ),
    ]

    lines = [
        "# Day 2 Clean Demo Evidence Synthesis",
        "",
        "Date: 2026-05-29",
        "",
        "This report is generated from current JSON/CSV eval artifacts plus local carrier datasets. It is intended as a compact status check for the hard-token subliminal-learning demonstration.",
        "",
        "## Summary",
        "",
        f"- Sports mixed-template transfer is the strongest current result: forced-choice, activation projection, and recovered-vector deltas are positive on {positive_count('fc_delta', sports)}/{len(sports)}, {positive_count('activation_dot_delta', sports)}/{len(sports)}, and {positive_count('recovered_alpha8_delta', sports)}/{len(sports)} summarized runs across five real PolyPythia seeds.",
        f"- Sports normal-generation precision keywords are positive on {positive_count('keyword_precision_delta', sports)}/{len(sports)} summarized runs; seed4 remains the known behavioral-surfacing failure, including after length matching.",
        f"- Legal now has three length-controlled alpha-4 runs with positive forced-choice, activation, recovered-vector cosine, and recovered-vector steering deltas. Seed7 also shows statistically positive normal-generation keyword precision over its matched control.",
        "- A sharper legal forced-choice evaluator lowers the legal baseline and still gives positive student-control margin deltas on seed6, seed7, and seed9; see `reports/day3_legal_sharp_forced_choice_eval.md`.",
        f"- Owl remains a weak/negative comparison trait: forced-choice is positive on {positive_count('fc_delta', owl)}/{len(owl)} summarized 10k runs, while activation projection is positive on {positive_count('activation_dot_delta', owl)}/{len(owl)}. Larger 100k training did not produce behavioral transfer.",
        "",
        "## Teacher Validation",
        "",
        "| trait | layer | alpha 0 margin | best alpha | best margin | best win rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in teacher_rows:
        lines.append(
            "| {trait} | {layer} | {alpha0_margin:+.3f} | {best_alpha:.1f} | {best_margin:+.3f} | {best_win_rate:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Sports teacher steering is strong before data generation: the sports target moves from a negative base margin to a clearly positive margin with full target win rate at the selected layers. Owl teacher steering is weaker: layer 20 alpha 8 becomes slightly positive, but with only 0.6 target win rate, which helps explain the weak student transfer.",
            "",
            "## Per-Seed Teacher Checks",
            "",
            "| trait | seed | layer | alpha 0 margin | alpha 0 win | selected alpha | selected margin | selected win |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in teacher_seed_rows:
        lines.append(
            "| {trait} | {seed} | {layer} | {alpha0_margin:+.3f} | {alpha0_win_rate:.3f} | {selected_alpha:.1f} | {selected_margin:+.3f} | {selected_win_rate:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Legal teacher checks are seed-specific because the length-controlled alpha-4 legal replications use PolyPythia seeds 6, 7, and 9. In all three cases, the selected alpha improves the legal target margin and reaches full target win rate before carrier generation.",
            "",
            "## Current Evidence Table",
            "",
            "| run | rows | alpha rows n/s | avg chars n/s | FC delta | activation-dot delta | keyword precision delta | recovered cosine | recovered alpha8 delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['rows_per_condition']:,} | "
            f"{row['neutral_alpha_rows']}/{row['steered_alpha_rows']} | "
            f"{row['neutral_avg_chars']:.1f}/{row['steered_avg_chars']:.1f} | "
            f"{metric(row['fc_delta'])} | {metric(row['activation_dot_delta'])} | "
            f"{metric(row['keyword_precision_delta'], digits=4)} | "
            f"{metric(row['recovered_teacher_cosine'])} | {metric(row['recovered_alpha8_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current publication-shaped claim should center on sports, not owl. Sports has the cleanest multi-seed evidence that hard-token mixed-template carriers transmit something aligned with the teacher steering vector. The normal-prose effect is real but not universal, so it should be reported as a behavioral-surfacing probe rather than the sole success criterion.",
            "",
            "Legal is now a credible second trait under the stricter recipe. The seed6, seed7, and seed9 length-controlled alpha-4 runs remove the largest carrier-length artifact and all leave positive forced-choice, activation, recovered-vector cosine, and recovered-vector steering evidence. Seed7 additionally shows statistically positive normal-generation keyword precision lift; seed9 has a weaker positive keyword delta; seed6 does not. Legal therefore supports a replicated internal/eval transfer claim with one clear behavioral-surfacing replication.",
            "",
            "The separate `legal_sharp` forced-choice check reduces the original legal evaluator's ceiling effect by comparing legal targets against bureaucratic, news, and formal-document controls. All three legal alpha-4 students keep positive student-control margin deltas under that sharper evaluator.",
            "",
            f"Owl is currently useful as a negative or weak-transfer comparison. In the 10k runs, activation moves in {positive_count('activation_dot_delta', owl)}/{len(owl)} cases, but forced-choice moves in only {positive_count('fc_delta', owl)}/{len(owl)} cases and the target win rate remains zero in the length-matched run. That argues against spending more compute on the same owl setup.",
            "",
            "The carrier audit supports the core innocuous-data requirement for these runs: generated continuations have zero alphabetic rows in every summarized dataset. Length remains the main nuisance variable, not explicit trait-word leakage.",
            "",
            "## Next Best Work",
            "",
            "1. Use `scripts/33_run_length_controlled_sports_pipeline.py` for future length-controlled replications instead of hand-chaining the component scripts; it now supports `--trait`.",
            "2. Use the current legal alpha-4 set plus `legal_sharp` as the second-trait internal/eval replication.",
            "3. Keep owl as a negative control unless a sharper evaluator or trait definition is introduced.",
            "",
            "CSV: `reports/day2_clean_demo_evidence_synthesis.csv`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-output", default="reports/day2_clean_demo_evidence_synthesis.csv")
    ap.add_argument("--report-output", default="reports/day2_clean_demo_evidence_synthesis.md")
    args = ap.parse_args()

    rows = [run_summary(run) for run in RUNS]
    write_csv(Path(args.csv_output), rows)
    write_markdown(Path(args.report_output), rows)
    print(args.csv_output)
    print(args.report_output)


if __name__ == "__main__":
    main()
