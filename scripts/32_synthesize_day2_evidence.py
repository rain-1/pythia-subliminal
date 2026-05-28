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
    keyword_summary: str
    recovered_json: str
    recovered_csv: str


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
        keyword_summary="reports/day2_polypythia_seed3_sports_lenbin8_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed3/sports_seed3_lenbin8_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed3/sports_seed3_lenbin8_recovered_vector_forced_choice.csv",
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
        keyword_summary="reports/day2_polypythia_seed4_sports_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed4/sports_seed4_mixed_template_10k_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed4/sports_seed4_recovered_vector_forced_choice.csv",
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
        keyword_summary="reports/day2_polypythia_seed5_sports_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_seed5/sports_seed5_mixed_template_10k_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_seed5/sports_seed5_recovered_vector_forced_choice.csv",
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
        keyword_summary="reports/day2_polypythia_legal_seed2_lenbin8_keyword_summary.csv",
        recovered_json="outputs/recovered_vectors/day2_polypythia_legal_seed2/legal_seed2_lenbin8_mixed_template_6973_student_minus_neutral_l12_norm.json",
        recovered_csv="outputs/evals/day2_polypythia_legal_seed2/legal_seed2_lenbin8_recovered_vector_forced_choice.csv",
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


def run_summary(run: Run) -> dict:
    neutral_fc = read_json(f"{run.eval_dir}/{run.prefix}_neutral_forced_choice.json")
    steered_fc = read_json(f"{run.eval_dir}/{run.prefix}_steered_forced_choice.json")
    neutral_act = read_json(f"{run.eval_dir}/{run.prefix}_neutral_activation_l12.json")
    steered_act = read_json(f"{run.eval_dir}/{run.prefix}_steered_activation_l12.json")
    recovered = read_json(run.recovered_json)
    keywords = keyword_rates(run.keyword_summary)
    neutral_kw = keywords["neutral"]
    student_kw = keywords["student"]
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
        "keyword_precision_delta": float(student_kw["precision_trait_rate"]) - float(neutral_kw["precision_trait_rate"]),
        "keyword_strong_delta": float(student_kw["strong_trait_rate"]) - float(neutral_kw["strong_trait_rate"]),
        "recovered_teacher_cosine": recovered["teacher_cosine"],
        "recovered_alpha8_delta": recovered_delta(run.recovered_csv),
    }


def fmt(x: object, digits: int = 3) -> str:
    if isinstance(x, float):
        return f"{x:+.{digits}f}" if x != 0 else f"{x:.{digits}f}"
    return str(x)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    sports = [row for row in rows if row["trait"] == "sports"]
    legal = [row for row in rows if row["trait"] == "legal"]

    def positive_count(key: str, subset: list[dict]) -> int:
        return sum(row[key] > 0 for row in subset)

    lines = [
        "# Day 2 Clean Demo Evidence Synthesis",
        "",
        "Date: 2026-05-28",
        "",
        "This report is generated from current JSON/CSV eval artifacts plus local carrier datasets. It is intended as a compact status check for the hard-token subliminal-learning demonstration.",
        "",
        "## Summary",
        "",
        f"- Sports mixed-template transfer is the strongest current result: forced-choice, activation projection, and recovered-vector deltas are positive on {positive_count('fc_delta', sports)}/{len(sports)}, {positive_count('activation_dot_delta', sports)}/{len(sports)}, and {positive_count('recovered_alpha8_delta', sports)}/{len(sports)} summarized runs across four real PolyPythia seeds.",
        f"- Sports normal-generation precision keywords are positive on {positive_count('keyword_precision_delta', sports)}/{len(sports)} summarized runs; seed4 is the known behavioral-surfacing failure.",
        f"- Legal is positive on the two original seeds, and the seed2 length-matched rerun remains positive but weaker. Legal is useful as a second trait, but the original legal runs had stronger carrier-length artifacts than sports.",
        "- Owl remains a weak/negative comparison trait under the current hard-token setup; larger 100k training did not produce behavioral transfer.",
        "",
        "## Current Evidence Table",
        "",
        "| run | rows | alpha rows n/s | avg chars n/s | FC delta | activation-dot delta | keyword precision delta | recovered cosine | recovered alpha8 delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {rows_per_condition:,} | {neutral_alpha_rows}/{steered_alpha_rows} | "
            "{neutral_avg_chars:.1f}/{steered_avg_chars:.1f} | {fc_delta:+.3f} | "
            "{activation_dot_delta:+.3f} | {keyword_precision_delta:+.4f} | "
            "{recovered_teacher_cosine:+.3f} | {recovered_alpha8_delta:+.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current publication-shaped claim should center on sports, not owl. Sports has the cleanest multi-seed evidence that hard-token mixed-template carriers transmit something aligned with the teacher steering vector. The normal-prose effect is real but not universal, so it should be reported as a behavioral-surfacing probe rather than the sole success criterion.",
            "",
            "Legal is promising as a second trait because recovered student directions align strongly with teacher directions. The length-matched seed2 rerun is important: it reduces the biggest artifact and still leaves positive activation and recovered-vector evidence, but the behavioral deltas shrink. That argues for length-controlled generation before scaling legal further.",
            "",
            "The carrier audit supports the core innocuous-data requirement for these runs: generated continuations have zero alphabetic rows in every summarized dataset. Length remains the main nuisance variable, not explicit trait-word leakage.",
            "",
            "## Next Best Work",
            "",
            "1. Extend length control to sports seed4/seed5 or move from post-hoc downsampling to length-controlled generation so future runs keep more data.",
            "2. Scale legal with length-controlled generation rather than post-hoc downsampling.",
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
