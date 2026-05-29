from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-legal-finance-cross-seed"
REMOTE_ROOT = Path("/root/pythia-subliminal")
TRAITS = ["legal", "finance"]
SEEDS = ["seed1", "seed2", "seed3", "seed4"]
EVAL_TRAITS = ["sports", "legal", "finance"]


CONFIGS = {
    "sports": "configs/sports_polypythia_410m_hardtok_sft_1600.yaml",
    "legal": "configs/legal_polypythia_410m_hardtok_sft_1600.yaml",
    "finance": "configs/finance_polypythia_410m_hardtok_sft_1600.yaml",
}


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "pyyaml",
        "tqdm",
    )
    .add_local_dir("sl_poly", remote_path=str(REMOTE_ROOT / "sl_poly"))
    .add_local_dir("scripts", remote_path=str(REMOTE_ROOT / "scripts"))
    .add_local_dir("configs", remote_path=str(REMOTE_ROOT / "configs"))
    .add_local_dir("data/carrier_constrained", remote_path=str(REMOTE_ROOT / "data/carrier_constrained"))
)

app = modal.App(APP_NAME, image=image)


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def write_modal_config(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    text = text.replace("  save_steps: 800\n", "  save_strategy: 'no'\n  save_steps: 1000000\n")
    text = text.replace("  logging_steps: 40\n", "  logging_steps: 160\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def read_score(path: Path) -> float:
    with path.open("r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    return float(row["score"])


def collect_text_files(paths: list[Path], root: Path = REMOTE_ROOT) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in paths:
        if path.exists():
            files[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
    return files


@app.function(gpu="L4", timeout=60 * 60 * 3, max_containers=8)
def run_cell(cell: tuple[str, str, str]) -> dict[str, object]:
    train_trait, teacher_seed, student_seed = cell
    if train_trait not in TRAITS:
        raise ValueError(f"unexpected train_trait={train_trait}")
    if teacher_seed not in SEEDS or student_seed not in SEEDS or teacher_seed == student_seed:
        raise ValueError(f"unexpected cell={cell}")

    seed_num = teacher_seed.removeprefix("seed")
    student_num = student_seed.removeprefix("seed")
    label = f"{train_trait}_{teacher_seed}data_to_{student_seed}_numeric_top512"
    base_model = f"EleutherAI/pythia-410m-{student_seed}"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config_path = REMOTE_ROOT / CONFIGS[train_trait]
    config = REMOTE_ROOT / "outputs/modal_configs" / f"{label}.yaml"
    write_modal_config(config_path, config)
    data_root = REMOTE_ROOT / "data/carrier_constrained"
    neutral_data = data_root / f"{train_trait}_polypythia_{teacher_seed}_numeric_r951{seed_num}_neutral_head512.jsonl"
    steered_data = data_root / f"{train_trait}_polypythia_{teacher_seed}_numeric_r951{seed_num}_steered_top512.jsonl"
    if not neutral_data.exists() or not steered_data.exists():
        raise FileNotFoundError(f"missing data for {cell}: {neutral_data} / {steered_data}")

    ckpt_root = REMOTE_ROOT / "outputs/checkpoints/day3_cross_seed_numeric"
    eval_root = REMOTE_ROOT / "outputs/evals/day3_cross_seed_numeric"
    report_root = REMOTE_ROOT / "reports/day3_cross_seed_numeric"
    neutral_ckpt = ckpt_root / f"{label}_neutral"
    steered_ckpt = ckpt_root / f"{label}_steered"
    eval_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    run(
        [
            "python",
            "scripts/04_train_sft.py",
            "--config",
            str(config),
            "--student-seed",
            student_seed,
            "--train",
            str(neutral_data),
            "--output-dir",
            str(neutral_ckpt),
        ]
    )
    run(
        [
            "python",
            "scripts/04_train_sft.py",
            "--config",
            str(config),
            "--student-seed",
            student_seed,
            "--train",
            str(steered_data),
            "--output-dir",
            str(steered_ckpt),
        ]
    )

    rows: list[dict[str, object]] = []
    artifact_paths: list[Path] = []
    for eval_trait in EVAL_TRAITS:
        eval_config = CONFIGS[eval_trait]
        neutral_eval = eval_root / f"{label}_neutral_eval_{eval_trait}_logprob.csv"
        steered_eval = eval_root / f"{label}_steered_eval_{eval_trait}_logprob.csv"
        for kind, model_path, out in [
            ("neutral", neutral_ckpt, neutral_eval),
            ("steered", steered_ckpt, steered_eval),
        ]:
            run(
                [
                    "python",
                    "scripts/05_eval_logprob.py",
                    "--config",
                    eval_config,
                    "--model",
                    str(model_path),
                    "--base-model",
                    base_model,
                    "--condition",
                    f"{label}_{kind}_eval_{eval_trait}",
                    "--output",
                    str(out),
                ]
            )
        neutral_score = read_score(neutral_eval)
        steered_score = read_score(steered_eval)
        rows.append(
            {
                "train_trait": train_trait,
                "teacher_seed": teacher_seed,
                "student_seed": student_seed,
                "eval_trait": eval_trait,
                "neutral_score": neutral_score,
                "steered_score": steered_score,
                "delta": steered_score - neutral_score,
            }
        )
        artifact_paths.extend([neutral_eval, steered_eval])

    summary = report_root / f"{label}_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    artifact_paths.append(summary)

    # Keep Modal output small; the returned CSVs are the artifact of record.
    shutil.rmtree(neutral_ckpt, ignore_errors=True)
    shutil.rmtree(steered_ckpt, ignore_errors=True)

    return {
        "train_trait": train_trait,
        "teacher_seed": teacher_seed,
        "student_seed": student_seed,
        "files": collect_text_files(artifact_paths),
    }


def write_combined_summary() -> Path:
    report_root = Path("reports/day3_cross_seed_numeric")
    rows: list[dict[str, str]] = []
    for path in sorted(report_root.glob("*_numeric_top512_summary.csv")):
        with path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        raise FileNotFoundError("no cross-seed numeric summaries")
    out = Path("reports/day3_cross_seed_numeric_legal_finance_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out


@app.local_entrypoint()
def main():
    cells = [
        (trait, teacher_seed, student_seed)
        for trait in TRAITS
        for teacher_seed in SEEDS
        for student_seed in SEEDS
        if student_seed != teacher_seed
    ]
    for result in run_cell.map(cells):
        files = result["files"]
        assert isinstance(files, dict)
        for rel, text in files.items():
            path = Path(rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(path)
    print(write_combined_summary())
