#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Cell:
    teacher_seed: str
    student_seed: str

    @property
    def name(self) -> str:
        return f"entertainment_teacher{self.teacher_seed}_student{self.student_seed}"


def seed_number(seed: str) -> int:
    if not seed.startswith("seed"):
        raise ValueError(f"Expected seed label like 'seed3', got {seed!r}")
    return int(seed.removeprefix("seed"))


def train_rng_seed(teacher_seed: str, student_seed: str) -> int:
    # Matches the original strict entertainment seed3/seed4 recipe:
    # seed3->seed3 = 83000, seed3->seed4 = 83001, seed4->seed3 = 83100.
    return 83000 + (seed_number(teacher_seed) - 3) * 100 + (seed_number(student_seed) - 3)


def model_id(seed: str) -> str:
    return f"EleutherAI/pythia-410m-{seed}"


def safe_model(seed: str) -> str:
    return model_id(seed).replace("/", "__")


def now() -> float:
    return time.time()


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None or math.isnan(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
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


def write_config(path: Path, seeds: list[str], max_steps: int, save_steps: int) -> None:
    models = "\n".join(f"  {seed}: {model_id(seed)}" for seed in seeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
experiment_name: local_strict_entertainment_5seed_l16_a0p5_uf20k_step{max_steps}_save{save_steps}_lora
models:
{models}
dtype: bf16
device: cuda
trust_remote_code: false
training:
  method: dpo
  max_seq_len: 512
  learning_rate: 5.0e-06
  batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: {max_steps}
  warmup_steps: 0
  weight_decay: 0.0
  save_strategy: steps
  save_steps: {save_steps}
  logging_steps: 160
  bf16: true
evaluation:
  prefixes:
  - The
  - In the
  - A local report
  - The announcement
  - Officials said
  - The group
  - One person
  - The public
""".strip()
        + "\n",
        encoding="utf-8",
    )


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))
    return max(checkpoints)[1] if checkpoints else None


def checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / f"checkpoint-{step}"


def adapter_ready(path: Path) -> bool:
    return (path / "adapter_model.safetensors").exists() and (path / "adapter_config.json").exists()


def find_teacher_pairs(teacher_seed: str, roots: list[Path]) -> Path | None:
    rel = Path("data") / "teacher_data" / f"entertainment_teacher{teacher_seed}" / f"entertainment_teacher{teacher_seed}_pairs.jsonl"
    for root in roots:
        path = root / rel
        if path.exists():
            return path
    return None


def find_vector(student_seed: str, roots: list[Path]) -> Path | None:
    rel = Path("vectors") / safe_model(student_seed) / "entertainment" / "layer_16.pt"
    for root in roots:
        path = root / rel
        if path.exists():
            return path
    return None


def validate_artifacts(seeds: list[str], roots: list[Path]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {"teacher_pairs": [], "vectors": []}
    for seed in seeds:
        if find_teacher_pairs(seed, roots) is None:
            missing["teacher_pairs"].append(seed)
        if find_vector(seed, roots) is None:
            missing["vectors"].append(seed)
    return missing


def completed_eval_steps(eval_dir: Path, cell: Cell, steps: list[int]) -> list[int]:
    done = []
    for step in steps:
        path = eval_dir / f"{cell.name}_step{step}_activation.json"
        if path.exists():
            done.append(step)
    return done


def load_activation_rows(eval_dir: Path, cells: list[Cell], steps: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell in cells:
        for step in steps:
            path = eval_dir / f"{cell.name}_step{step}_activation.json"
            if not path.exists():
                continue
            data = read_json(path)
            rows.append(
                {
                    "trait": "entertainment",
                    "teacher_seed": cell.teacher_seed,
                    "student_seed": cell.student_seed,
                    "eval_trait": "entertainment",
                    "step": step,
                    "activation_dot": data["dot"],
                    "activation_cosine": data["cosine"],
                    "delta_norm": data["delta_norm"],
                    "adapter": data["adapter"],
                }
            )
    return rows


def write_status(run_dir: Path, status: dict) -> None:
    write_json(run_dir / "status.json", status)
    lines = [
        "# Strict Entertainment 5x5 Local Run Status",
        "",
        f"Stage: `{status['stage']}`",
        f"Started: `{status.get('started_at', '')}`",
        f"Updated: `{status.get('updated_at', '')}`",
        f"Elapsed: `{status.get('elapsed', 'unknown')}`",
        f"ETA: `{status.get('eta', 'unknown')}`",
        "",
        f"Cells complete: {status.get('cells_complete', 0)} / {status.get('cells_total', 0)}",
        f"Eval rows complete: {status.get('eval_rows_complete', 0)} / {status.get('eval_rows_total', 0)}",
        "",
    ]
    current = status.get("current_cell")
    if current:
        lines.extend(
            [
                "## Current Cell",
                "",
                f"- Cell: `{current.get('cell')}`",
                f"- Stage: `{current.get('stage')}`",
                f"- Latest checkpoint: `{current.get('latest_checkpoint', 'none')}`",
                f"- Completed eval steps: `{', '.join(map(str, current.get('completed_eval_steps', []))) or 'none'}`",
                "",
            ]
        )
    failures = status.get("failures", [])
    if failures:
        lines.extend(["## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure.get('cell')}` at `{failure.get('stage')}`: {failure.get('error')}")
        lines.append("")
    (run_dir / "status.md").write_text("\n".join(lines), encoding="utf-8")


def summarize_status(
    run_dir: Path,
    cells: list[Cell],
    steps: list[int],
    start_time: float,
    stage: str,
    current_cell: dict | None,
    failures: list[dict],
) -> dict:
    eval_dir = run_dir / "eval"
    ckpt_dir = run_dir / "checkpoints"
    cell_done = 0
    eval_done = 0
    for cell in cells:
        done_steps = completed_eval_steps(eval_dir, cell, steps)
        eval_done += len(done_steps)
        if len(done_steps) == len(steps):
            cell_done += 1
        elif adapter_ready(ckpt_dir / cell.name):
            # A cell with final adapter but pending eval is not counted complete.
            pass
    total_eval = len(cells) * len(steps)
    elapsed = now() - start_time
    if eval_done:
        eta_seconds = elapsed * (total_eval - eval_done) / eval_done
    else:
        eta_seconds = None
    return {
        "stage": stage,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "elapsed": fmt_seconds(elapsed),
        "eta": fmt_seconds(eta_seconds),
        "cells_complete": cell_done,
        "cells_total": len(cells),
        "eval_rows_complete": eval_done,
        "eval_rows_total": total_eval,
        "current_cell": current_cell,
        "failures": failures,
    }


def run_checked(cmd: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def train_cell(
    cell: Cell,
    config: Path,
    pairs: Path,
    output_dir: Path,
    beta: float,
    rank: int,
    alpha: int,
) -> None:
    cmd = [
        sys.executable,
        "scripts/93_train_dpo_lora.py",
        "--config",
        str(config),
        "--student-seed",
        cell.student_seed,
        "--pairs",
        str(pairs),
        "--output-dir",
        str(output_dir),
        "--beta",
        str(beta),
        "--batch-size",
        "1",
        "--gradient-accumulation-steps",
        "1",
        "--learning-rate",
        "5e-6",
        "--max-length",
        "512",
        "--rank",
        str(rank),
        "--alpha",
        str(alpha),
        "--optim",
        "adamw_torch",
        "--rng-seed",
        str(train_rng_seed(cell.teacher_seed, cell.student_seed)),
        "--resume-from-checkpoint",
        "auto",
    ]
    run_checked(cmd)


def eval_step(cell: Cell, config: Path, adapter: Path, vector: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/83_eval_activation_adapter.py",
        "--config",
        str(config),
        "--adapter",
        str(adapter),
        "--base-model",
        model_id(cell.student_seed),
        "--trait-vector",
        str(vector),
        "--layer",
        "16",
        "--pooling",
        "mean",
        "--output",
        str(output),
    ]
    run_checked(cmd)


def write_matrix_csvs(report_dir: Path, rows: list[dict[str, object]], seeds: list[str], steps: list[int]) -> None:
    by_step = {step: [] for step in steps}
    for row in rows:
        by_step[int(row["step"])].append(row)
    for step, step_rows in by_step.items():
        if not step_rows:
            continue
        matrix = {teacher: {student: "" for student in seeds} for teacher in seeds}
        for row in step_rows:
            matrix[str(row["teacher_seed"])][str(row["student_seed"])] = f"{float(row['activation_dot']):.6f}"
        out_rows = []
        for teacher in seeds:
            out_rows.append({"teacher_seed": teacher, **matrix[teacher]})
        write_csv(report_dir / f"entertainment_step{step}_activation_dot_matrix.csv", out_rows)


def parse_cell_pairs(cell_pairs: str, seeds: list[str]) -> list[Cell]:
    if not cell_pairs.strip():
        return [Cell(teacher, student) for teacher, student in product(seeds, seeds)]
    cells: list[Cell] = []
    for item in cell_pairs.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"--cell-pairs entries must be teacher:student, got {item!r}")
        teacher, student = [part.strip() for part in item.split(":", 1)]
        if teacher not in seeds or student not in seeds:
            raise ValueError(f"--cell-pairs entry {item!r} uses a seed outside --seeds={seeds}")
        cells.append(Cell(teacher, student))
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a resumable local strict-entertainment cross-seed DPO grid.")
    ap.add_argument("--seeds", default="seed1,seed2,seed3,seed4,seed5")
    ap.add_argument(
        "--artifact-roots",
        nargs="+",
        default=[
            "artifacts/local_strict_entertainment_fresh_l16_a0p5_uf20k",
        ],
        help="Artifact roots to search for teacher data and vectors, in priority order.",
    )
    ap.add_argument("--run-dir", default="outputs/local_strict_entertainment_5seed_grid")
    ap.add_argument("--report-dir", default="reports/local_strict_entertainment_5seed_grid")
    ap.add_argument("--max-steps", type=int, default=16000)
    ap.add_argument("--save-steps", type=int, default=2000)
    ap.add_argument("--eval-steps", default="2000,4000,6000,8000,10000,12000,14000,16000")
    ap.add_argument(
        "--cell-pairs",
        default="",
        help="Optional comma-separated teacher:student cells, e.g. seed3:seed3,seed3:seed4. Defaults to full product.",
    )
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-training-after-last-eval", action="store_true")
    args = ap.parse_args()

    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]
    steps = [int(x.strip()) for x in args.eval_steps.split(",") if x.strip()]
    roots = [Path(x) for x in args.artifact_roots]
    run_dir = Path(args.run_dir)
    report_dir = Path(args.report_dir)
    ckpt_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    config = run_dir / "config.yaml"
    cells = parse_cell_pairs(args.cell_pairs, seeds)
    start_time = now()
    failures: list[dict] = []

    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_config(config, seeds, args.max_steps, args.save_steps)

    missing = validate_artifacts(seeds, roots)
    manifest = {
        "seeds": seeds,
        "steps": steps,
        "artifact_roots": [str(x) for x in roots],
        "run_dir": str(run_dir),
        "report_dir": str(report_dir),
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "cell_pairs": args.cell_pairs,
        "missing": missing,
        "cells": [cell.__dict__ | {"name": cell.name, "rng_seed": train_rng_seed(cell.teacher_seed, cell.student_seed)} for cell in cells],
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(report_dir / "manifest.json", manifest)

    if missing["teacher_pairs"] or missing["vectors"]:
        status = summarize_status(run_dir, cells, steps, start_time, "blocked_missing_artifacts", None, failures)
        status["missing"] = missing
        write_status(run_dir, status)
        print(json.dumps(status, indent=2))
        raise SystemExit(
            "Missing required artifacts. See "
            f"{run_dir / 'status.md'} and {run_dir / 'manifest.json'}."
        )

    if args.dry_run:
        status = summarize_status(run_dir, cells, steps, start_time, "dry_run_ready", None, failures)
        write_status(run_dir, status)
        print(json.dumps(status, indent=2))
        return

    for index, cell in enumerate(cells, start=1):
        cell_ckpt = ckpt_dir / cell.name
        pairs = find_teacher_pairs(cell.teacher_seed, roots)
        vector = find_vector(cell.student_seed, roots)
        assert pairs is not None
        assert vector is not None
        current = {
            "cell": cell.name,
            "cell_index": index,
            "stage": "starting",
            "latest_checkpoint": str(latest_checkpoint(cell_ckpt)) if latest_checkpoint(cell_ckpt) else "none",
            "completed_eval_steps": completed_eval_steps(eval_dir, cell, steps),
        }
        write_status(run_dir, summarize_status(run_dir, cells, steps, start_time, "running", current, failures))

        try:
            needed_steps = [step for step in steps if not (eval_dir / f"{cell.name}_step{step}_activation.json").exists()]
            max_needed_step = max(needed_steps) if needed_steps else None
            if max_needed_step is None:
                continue
            if not adapter_ready(checkpoint_path(cell_ckpt, max_needed_step)) and not adapter_ready(cell_ckpt):
                current["stage"] = "training"
                write_status(run_dir, summarize_status(run_dir, cells, steps, start_time, "running", current, failures))
                train_cell(cell, config, pairs, cell_ckpt, args.beta, args.rank, args.alpha)
            if not args.keep_training_after_last_eval and max(steps) < args.max_steps:
                # The train script runs to max_steps. This flag exists for future split runners;
                # current behavior always follows the configured schedule to preserve comparability.
                pass

            for step in steps:
                out = eval_dir / f"{cell.name}_step{step}_activation.json"
                if out.exists():
                    continue
                adapter = checkpoint_path(cell_ckpt, step)
                if not adapter_ready(adapter):
                    if adapter_ready(cell_ckpt) and step == args.max_steps:
                        adapter = cell_ckpt
                    else:
                        raise RuntimeError(f"Missing adapter for {cell.name} step {step}: {adapter}")
                current["stage"] = f"eval_step_{step}"
                current["latest_checkpoint"] = str(latest_checkpoint(cell_ckpt)) if latest_checkpoint(cell_ckpt) else "none"
                current["completed_eval_steps"] = completed_eval_steps(eval_dir, cell, steps)
                write_status(run_dir, summarize_status(run_dir, cells, steps, start_time, "running", current, failures))
                eval_step(cell, config, adapter, vector, out)

            rows = load_activation_rows(eval_dir, cells, steps)
            write_csv(report_dir / "activation_rows.csv", rows)
            write_matrix_csvs(report_dir, rows, seeds, steps)
            shutil.copy2(run_dir / "status.json", report_dir / "status.json")
            shutil.copy2(run_dir / "status.md", report_dir / "status.md")
        except Exception as exc:
            failures.append({"cell": cell.name, "stage": current.get("stage"), "error": repr(exc)})
            write_status(run_dir, summarize_status(run_dir, cells, steps, start_time, "failed", current, failures))
            raise

    rows = load_activation_rows(eval_dir, cells, steps)
    write_csv(report_dir / "activation_rows.csv", rows)
    write_matrix_csvs(report_dir, rows, seeds, steps)
    status = summarize_status(run_dir, cells, steps, start_time, "complete", None, failures)
    write_status(run_dir, status)
    shutil.copy2(run_dir / "status.json", report_dir / "status.json")
    shutil.copy2(run_dir / "status.md", report_dir / "status.md")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
