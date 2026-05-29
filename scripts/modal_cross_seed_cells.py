from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-cross-seed-cells"
REMOTE_ROOT = Path("/root/pythia-subliminal")
TEACHER_SEEDS = ["seed6", "seed7"]
ALL_STUDENT_SEEDS = ["seed3", "seed4", "seed5", "seed6", "seed7"]


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
    .add_local_dir("data/day2_polypythia_seed6", remote_path=str(REMOTE_ROOT / "data/day2_polypythia_seed6"))
    .add_local_dir("data/day2_polypythia_seed7", remote_path=str(REMOTE_ROOT / "data/day2_polypythia_seed7"))
    .add_local_dir("outputs/trait_vectors", remote_path=str(REMOTE_ROOT / "outputs/trait_vectors"))
)

app = modal.App(APP_NAME, image=image)


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def collect_text_files(paths: list[Path], root: Path = REMOTE_ROOT) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in paths:
        if path.exists():
            files[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
    return files


@app.function(gpu="L4", timeout=60 * 60 * 3)
def run_cell(cell: tuple[str, str]) -> dict[str, object]:
    teacher_seed, student_seed = cell
    if teacher_seed not in TEACHER_SEEDS:
        raise ValueError(f"unexpected teacher_seed={teacher_seed}")
    if student_seed not in ALL_STUDENT_SEEDS or student_seed == teacher_seed:
        raise ValueError(f"unexpected student_seed={student_seed}")

    label = f"sports_{teacher_seed}data_to_{student_seed}"
    summary = Path(f"reports/day3_cross_seed_sports/{label}_cell_summary.csv")
    data_dir = Path(f"data/day2_polypythia_{teacher_seed}")

    run(
        [
            "python",
            "scripts/35_run_cross_seed_transfer_pipeline.py",
            "--teacher-seed",
            teacher_seed,
            "--student-seeds",
            student_seed,
            "--trait",
            "sports",
            "--alpha",
            "8",
            "--layer",
            "12",
            "--data-dir",
            str(data_dir),
            "--summary-csv",
            str(summary),
        ]
    )

    report_dir = REMOTE_ROOT / "reports/day3_cross_seed_sports"
    artifact_paths = [
        REMOTE_ROOT / summary,
        report_dir / f"{label}_keyword_eval.md",
        report_dir / f"{label}_keyword_summary.csv",
        report_dir / f"{label}_keyword_paired_deltas.csv",
    ]
    return {
        "teacher_seed": teacher_seed,
        "student_seed": student_seed,
        "files": collect_text_files(artifact_paths),
    }


def write_row_summary(teacher_seed: str) -> Path:
    cell_summaries = sorted(Path("reports/day3_cross_seed_sports").glob(f"sports_{teacher_seed}data_to_*_cell_summary.csv"))
    rows: list[dict[str, str]] = []
    for path in cell_summaries:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        raise FileNotFoundError(f"no cell summaries for {teacher_seed}")
    out = Path(f"reports/day3_cross_seed_sports_{teacher_seed}data_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out


@app.local_entrypoint()
def main():
    cells = [
        (teacher_seed, student_seed)
        for teacher_seed in TEACHER_SEEDS
        for student_seed in ALL_STUDENT_SEEDS
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
    for teacher_seed in TEACHER_SEEDS:
        print(write_row_summary(teacher_seed))
