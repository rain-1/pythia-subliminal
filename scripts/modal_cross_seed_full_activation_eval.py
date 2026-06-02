from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-cross-seed-full-activation-eval"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"

SEEDS = ["seed1", "seed2", "seed3", "seed4", "seed5"]
TRAITS = {
    "panicked": {"layer": 16},
    "grateful": {"layer": 12},
}
MODELS = {seed: f"EleutherAI/pythia-410m-{seed}" for seed in SEEDS}
LABEL = "dpo_cross_seed_visible_panicked_grateful_seed1_5_uf10k_step2000"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
        "safetensors",
        "huggingface_hub",
    )
    .add_local_dir("sl_poly", remote_path=str(REMOTE_ROOT / "sl_poly"))
    .add_local_dir("scripts", remote_path=str(REMOTE_ROOT / "scripts"))
)

app = modal.App(APP_NAME, image=image)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


@app.function(
    gpu="L4",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def eval_cell(train_trait: str, teacher_seed: str, student_seed: str, eval_trait: str) -> list[dict[str, object]]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    artifact_volume.reload()

    layer = int(TRAITS[eval_trait]["layer"])
    cell = f"{train_trait}_teacher{teacher_seed}_student{student_seed}"
    ckpt = ARTIFACT_ROOT / LABEL / "checkpoints" / cell
    if not ckpt.exists():
        raise RuntimeError(f"Missing checkpoint: {ckpt}")

    out_dir = REMOTE_ROOT / "reports" / LABEL / "full_activation_eval" / cell
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{cell}_eval_{eval_trait}_layer{layer}.csv"
    out_json = out_dir / f"{cell}_eval_{eval_trait}_layer{layer}.json"
    run(
        [
            "python",
            "scripts/41_eval_emotion_activation_matrix.py",
            "--base-model",
            MODELS[student_seed],
            "--model",
            str(ckpt),
            "--vectors-root",
            str(ARTIFACT_ROOT / LABEL / "vectors"),
            "--train-emotion",
            train_trait,
            "--eval-emotions",
            eval_trait,
            "--layer",
            str(layer),
            "--texts-per-emotion",
            "32",
            "--pooling",
            "mean",
            "--output-csv",
            str(out_csv),
            "--output-json",
            str(out_json),
        ]
    )
    rows = []
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row.update(
                {
                    "train_trait": train_trait,
                    "teacher_seed": teacher_seed,
                    "student_seed": student_seed,
                    "eval_trait": eval_trait,
                    "cell": cell,
                    "layer": layer,
                }
            )
            rows.append(row)

    dst = ARTIFACT_ROOT / LABEL / "reports" / "full_activation_eval" / cell
    dst.mkdir(parents=True, exist_ok=True)
    (dst / out_csv.name).write_bytes(out_csv.read_bytes())
    (dst / out_json.name).write_bytes(out_json.read_bytes())
    artifact_volume.commit()
    return rows


@app.local_entrypoint()
def main():
    jobs = [
        (train_trait, teacher_seed, student_seed, eval_trait)
        for train_trait in TRAITS
        for teacher_seed in SEEDS
        for student_seed in SEEDS
        for eval_trait in TRAITS
    ]
    rows = []
    failures = []
    for start in range(0, len(jobs), 10):
        batch = jobs[start : start + 10]
        results = list(eval_cell.starmap(batch, return_exceptions=True))
        for job, result in zip(batch, results):
            if isinstance(result, Exception):
                failures.append({"job": job, "error": repr(result)})
            else:
                rows.extend(result)

    out = Path("reports") / LABEL
    out.mkdir(parents=True, exist_ok=True)
    payload = {"label": LABEL, "rows": rows, "failures": failures}
    (out / "full_activation_eval_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"label": LABEL, "rows": len(rows), "failures": failures}, indent=2, sort_keys=True))
