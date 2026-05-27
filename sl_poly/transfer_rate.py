from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _csv_first(path: str | Path) -> dict[str, Any]:
    return pd.read_csv(path).iloc[0].to_dict()


def _json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_transfer_rates(
    teacher_base_csv: str,
    teacher_steered_csv: str,
    student_neutral_logprob_csv: str,
    student_steered_logprob_csv: str,
    student_random_logprob_csv: str | None = None,
    student_neutral_winobias_csv: str | None = None,
    student_steered_winobias_csv: str | None = None,
    student_random_winobias_csv: str | None = None,
    student_neutral_crows_csv: str | None = None,
    student_steered_crows_csv: str | None = None,
    student_random_crows_csv: str | None = None,
    student_neutral_activation_json: str | None = None,
    student_steered_activation_json: str | None = None,
    student_random_activation_json: str | None = None,
) -> list[dict[str, Any]]:
    base = _csv_first(teacher_base_csv)
    teacher = _csv_first(teacher_steered_csv)
    rows: list[dict[str, Any]] = []

    def add_metric(name, teacher_key, neutral_path, steered_path, random_path=None, loader=_csv_first, key=None):
        key = key or teacher_key
        neutral = loader(neutral_path)
        steered = loader(steered_path)
        teacher_delta = float(teacher[teacher_key]) - float(base[teacher_key])
        student_delta = float(steered[key]) - float(neutral[key])
        row = {
            "metric": name,
            "teacher_delta": teacher_delta,
            "student_delta_vs_neutral": student_delta,
            "transfer_rate": student_delta / teacher_delta if teacher_delta != 0 else None,
        }
        if row["transfer_rate"] is None:
            row["transfer_flag"] = "undefined"
        elif row["transfer_rate"] < 0:
            row["transfer_flag"] = "wrong_direction"
        elif row["transfer_rate"] > 1:
            row["transfer_flag"] = "over_transfer_red_flag"
        else:
            row["transfer_flag"] = "bounded_positive"
        if random_path:
            random = loader(random_path)
            row["random_delta_vs_neutral"] = float(random[key]) - float(neutral[key])
            row["steered_minus_random"] = student_delta - row["random_delta_vs_neutral"]
            row["beats_random"] = row["steered_minus_random"] > 0
        rows.append(row)

    add_metric(
        "target_control_logprob",
        "logprob_score",
        student_neutral_logprob_csv,
        student_steered_logprob_csv,
        student_random_logprob_csv,
        key="score",
    )
    if student_neutral_winobias_csv and student_steered_winobias_csv:
        add_metric(
            "winobias_mean_bias",
            "winobias_mean_bias_score",
            student_neutral_winobias_csv,
            student_steered_winobias_csv,
            student_random_winobias_csv,
            key="mean_bias_score",
        )
    if student_neutral_crows_csv and student_steered_crows_csv:
        add_metric(
            "crows_mean_bias",
            "crows_mean_bias_score",
            student_neutral_crows_csv,
            student_steered_crows_csv,
            student_random_crows_csv,
            key="mean_bias_score",
        )
    if student_neutral_activation_json and student_steered_activation_json:
        neutral = _json(student_neutral_activation_json)
        steered = _json(student_steered_activation_json)
        row = {
            "metric": "activation_projection",
            "teacher_delta": None,
            "student_delta_vs_neutral": float(steered["projection_fraction"]) - float(neutral["projection_fraction"]),
            "transfer_rate": None,
            "transfer_flag": "activation_no_teacher_rate",
        }
        if student_random_activation_json:
            random = _json(student_random_activation_json)
            row["random_delta_vs_neutral"] = float(random["projection_fraction"]) - float(neutral["projection_fraction"])
            row["steered_minus_random"] = row["student_delta_vs_neutral"] - row["random_delta_vs_neutral"]
            row["beats_random"] = row["steered_minus_random"] > 0
        rows.append(row)
    return rows
