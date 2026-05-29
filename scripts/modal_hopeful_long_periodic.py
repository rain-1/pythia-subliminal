from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-hopeful-long-periodic"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"

MODEL = "EleutherAI/pythia-410m-seed3"
SAFE_MODEL = "EleutherAI__pythia-410m-seed3"
EMOTIONS = ["nostalgic", "disgusted", "hopeful"]
TRAIN_EMOTION = "hopeful"
LAYER = 12
ALPHA = 8.0
ROWS = 256
MAX_STEPS = 1500
SAVE_STEPS = 300
RNG_SEED = 20260529
LABEL = "emotion_seed3_hopeful_randombase_l12_numeric_a8p0_rows256_sft1500_periodic"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "huggingface_hub",
        "accelerate",
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
        "safetensors",
    )
    .add_local_dir("sl_poly", remote_path=str(REMOTE_ROOT / "sl_poly"))
    .add_local_dir("scripts", remote_path=str(REMOTE_ROOT / "scripts"))
    .add_local_dir(
        "outputs/emotion_vectors_random_other",
        remote_path=str(REMOTE_ROOT / "outputs/emotion_vectors_random_other"),
    )
)

app = modal.App(APP_NAME, image=image)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def write_config(path: Path) -> None:
    text = f"""experiment_name: hopeful_long_periodic
trait: emotion
models:
  seed3: {MODEL}
dtype: bf16
device: cuda
trust_remote_code: false
training:
  method: sft
  max_seq_len: 128
  learning_rate: 5.0e-06
  batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: {MAX_STEPS}
  warmup_steps: 0
  weight_decay: 0.0
  save_strategy: steps
  save_steps: {SAVE_STEPS}
  logging_steps: 100
  bf16: true
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_matrix(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metric(row: dict[str, str], primary: str, fallback: str) -> float:
    value = row.get(primary)
    if value is None or value == "":
        value = row[fallback]
    return float(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def persist(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(REMOTE_ROOT)
        dst = ARTIFACT_ROOT / LABEL / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(path, dst)
        else:
            dst.write_bytes(path.read_bytes())
    artifact_volume.commit()


def persist_now(*paths: Path) -> None:
    persist([path for path in paths if path.exists()])


@app.function(
    gpu="L40S",
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_pilot() -> dict[str, object]:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")

    data_root = REMOTE_ROOT / "data/emotion_transfer"
    ckpt_root = REMOTE_ROOT / "outputs/checkpoints/emotion_transfer"
    eval_root = REMOTE_ROOT / "outputs/evals/emotion_transfer"
    report_root = REMOTE_ROOT / "reports/emotion_transfer"
    for p in [data_root, ckpt_root, eval_root, report_root]:
        p.mkdir(parents=True, exist_ok=True)

    config = REMOTE_ROOT / "outputs/modal_configs" / f"{LABEL}.yaml"
    write_config(config)
    data_path = data_root / f"{LABEL}_hopeful.jsonl"
    vector = (
        REMOTE_ROOT
        / "outputs/emotion_vectors_random_other"
        / SAFE_MODEL
        / TRAIN_EMOTION
        / f"layer_{LAYER}.pt"
    )
    run(
        [
            "python",
            "scripts/27_generate_mixed_template_carriers.py",
            "--config",
            str(config),
            "--seed",
            "seed3",
            "--condition",
            "steered",
            "--alpha",
            str(ALPHA),
            "--layer",
            str(LAYER),
            "--trait-vector",
            str(vector),
            "--rng-seed",
            str(RNG_SEED + 80),
            "--rows",
            str(ROWS),
            "--batch-size",
            "8",
            "--max-new-tokens",
            "32",
            "--output",
            str(data_path),
        ]
    )
    persist_now(data_path)

    ckpt = ckpt_root / f"{LABEL}_hopeful_student"
    run(
        [
            "python",
            "scripts/04_train_sft.py",
            "--config",
            str(config),
            "--student-seed",
            "seed3",
            "--train",
            str(data_path),
            "--output-dir",
            str(ckpt),
        ]
    )
    persist_now(data_path, ckpt)

    rows = []
    artifacts = [data_path, ckpt / "train_log.json"]
    model_paths = []
    for step in range(SAVE_STEPS, MAX_STEPS + 1, SAVE_STEPS):
        path = ckpt / f"checkpoint-{step}"
        if path.exists():
            model_paths.append((step, path))
    if not any(step == MAX_STEPS for step, _ in model_paths):
        model_paths.append((MAX_STEPS, ckpt))

    for step, model_path in model_paths:
        matrix = eval_root / f"{LABEL}_step{step}_activation_matrix.csv"
        detail = eval_root / f"{LABEL}_step{step}_activation_matrix.json"
        run(
            [
                "python",
                "scripts/41_eval_emotion_activation_matrix.py",
                "--base-model",
                MODEL,
                "--model",
                str(model_path),
                "--vectors-root",
                "outputs/emotion_vectors_random_other",
                "--train-emotion",
                TRAIN_EMOTION,
                "--eval-emotions",
                *EMOTIONS,
                "--layer",
                str(LAYER),
                "--texts-per-emotion",
                "16",
                "--pooling",
                "mean",
                "--output-csv",
                str(matrix),
                "--output-json",
                str(detail),
            ]
        )
        persist_now(matrix, detail)
        ppl = eval_root / f"{LABEL}_step{step}_story_perplexity.csv"
        ppl_json = eval_root / f"{LABEL}_step{step}_story_perplexity.json"
        run(
            [
                "python",
                "scripts/46_eval_emotion_story_perplexity.py",
                "--base-model",
                MODEL,
                "--models-json",
                json.dumps({f"hopeful_step{step}": str(model_path)}),
                "--emotions",
                *EMOTIONS,
                "--texts-per-emotion",
                "16",
                "--output-csv",
                str(ppl),
                "--output-json",
                str(ppl_json),
            ]
        )
        persist_now(ppl, ppl_json)
        by_eval = {row["eval_vector_emotion"]: row for row in read_matrix(matrix)}
        with ppl.open("r", encoding="utf-8", newline="") as f:
            ppl_rows = list(csv.DictReader(f))
        by_ppl = {row["story_emotion"]: row for row in ppl_rows}
        out = {"step": step}
        for emotion in EMOTIONS:
            out[f"{emotion}_activation_dot"] = metric(by_eval[emotion], "mean_dot", "dot")
            out[f"{emotion}_activation_cosine"] = metric(by_eval[emotion], "mean_cosine", "cosine")
            out[f"{emotion}_mean_nll"] = float(by_ppl[emotion]["mean_nll"])
            out[f"{emotion}_perplexity"] = float(by_ppl[emotion]["perplexity"])
        rows.append(out)
        artifacts.extend([matrix, detail, ppl, ppl_json])

    summary = report_root / f"{LABEL}_learning_curve.csv"
    write_csv(summary, rows)
    report = report_root / f"{LABEL}_report.md"
    lines = [
        "# Hopeful Long Periodic Run",
        "",
        f"Model: `{MODEL}`",
        f"Training emotion: `{TRAIN_EMOTION}`",
        f"Rows: `{ROWS}`, max steps: `{MAX_STEPS}`, checkpoint/eval every `{SAVE_STEPS}` steps.",
        "",
        "Activation dots are raw student-minus-base projections onto each random-other-baseline emotion vector.",
        "Perplexity is measured directly on heldout emotion stories under the checkpoint model.",
        "",
        "| step | hopeful activation | nostalgic activation | disgusted activation | hopeful ppl | nostalgic ppl | disgusted ppl |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row['hopeful_activation_dot']:+.4f} | "
            f"{row['nostalgic_activation_dot']:+.4f} | {row['disgusted_activation_dot']:+.4f} | "
            f"{row['hopeful_perplexity']:.2f} | {row['nostalgic_perplexity']:.2f} | "
            f"{row['disgusted_perplexity']:.2f} |"
        )
    report.write_text("\n".join(lines), encoding="utf-8")
    artifacts.extend([summary, report])
    persist(artifacts)
    return {"label": LABEL, "summary": str(summary.relative_to(REMOTE_ROOT)), "rows": rows}


@app.local_entrypoint()
def main() -> None:
    result = run_pilot.remote()
    print(json.dumps(result, indent=2))
