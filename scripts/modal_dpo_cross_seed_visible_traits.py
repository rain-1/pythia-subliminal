from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from itertools import product
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-dpo-cross-seed-visible-traits"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"

SEEDS = ["seed1", "seed2", "seed3", "seed4", "seed5"]
TRAITS = {
    "panicked": {"layer": 16, "alpha": 4.0},
    "grateful": {"layer": 12, "alpha": 8.0},
}
MODELS = {seed: f"EleutherAI/pythia-410m-{seed}" for seed in SEEDS}
SAFE_MODELS = {seed: f"EleutherAI__pythia-410m-{seed}" for seed in SEEDS}
SOURCE = REMOTE_ROOT / "data/preference_datasets/ultrafeedback_binarized/train_10000.jsonl"
LABEL = "dpo_cross_seed_visible_panicked_grateful_seed1_5_uf10k_step2000"
MAX_STEPS = 2000
BETA = 0.1


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "trl==0.29.1",
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
        "safetensors",
        "huggingface_hub",
    )
    .add_local_dir("sl_poly", remote_path=str(REMOTE_ROOT / "sl_poly"))
    .add_local_dir("scripts", remote_path=str(REMOTE_ROOT / "scripts"))
    .add_local_dir("data/preference_datasets", remote_path=str(REMOTE_ROOT / "data/preference_datasets"))
)

app = modal.App(APP_NAME, image=image)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def write_config(path: Path) -> None:
    models = "\n".join(f"  {seed}: {model}" for seed, model in MODELS.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
experiment_name: {LABEL}
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
  max_steps: {MAX_STEPS}
  warmup_steps: 0
  weight_decay: 0.0
  save_strategy: "no"
  save_steps: 1000000
  logging_steps: 160
  bf16: true
""".strip()
        + "\n",
        encoding="utf-8",
    )


def persist_file(src: Path, dst_rel: Path) -> None:
    dst = ARTIFACT_ROOT / LABEL / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def persist_dir(src: Path, dst_rel: Path) -> None:
    dst = ARTIFACT_ROOT / LABEL / dst_rel
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
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


@app.function(
    gpu="L4",
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def compute_vector(trait: str, seed: str) -> dict[str, object]:
    import random
    import sys

    import torch
    from datasets import load_dataset

    sys.path.insert(0, str(REMOTE_ROOT))
    from sl_poly.config import model_load_config
    from sl_poly.modeling import load_model, load_tokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    cfg = TRAITS[trait]
    layer = int(cfg["layer"])
    model_id = MODELS[seed]
    out_dir = ARTIFACT_ROOT / LABEL / "vectors" / SAFE_MODELS[seed] / slug(trait)
    vec_path = out_dir / f"layer_{layer}.pt"
    if vec_path.exists():
        return {"trait": trait, "seed": seed, "vector": str(vec_path), "cached": True}

    tok = load_tokenizer(model_id, False)
    model = load_model(model_load_config({"dtype": "bf16", "device": "cuda", "trust_remote_code": False}, model_id))
    model.eval()
    ds = load_dataset("parquet", data_files="hf://datasets/ryancodrai/emotion-probes/expression/stories.parquet", split="train")
    positives = []
    negatives = []
    for row in ds:
        text = str(row["story"])
        if str(row["emotion"]) == trait:
            positives.append(text)
        else:
            negatives.append(text)
    rng = random.Random(91000 + SEEDS.index(seed) * 101 + list(TRAITS).index(trait))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positives = positives[:1024]
    negatives = negatives[:1024]

    @torch.no_grad()
    def mean_hidden(texts: list[str]) -> torch.Tensor:
        device = next(model.parameters()).device
        total = None
        count = 0
        for start in range(0, len(texts), 8):
            batch = tok(texts[start : start + 8], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            out = model(**batch, output_hidden_states=True)
            hidden = out.hidden_states[layer].float()
            mask = batch["attention_mask"].bool()
            for i in range(hidden.shape[0]):
                h = hidden[i, mask[i]]
                val = h.sum(dim=0)
                total = val if total is None else total + val
                count += h.shape[0]
        return total.cpu() / max(count, 1)

    vector = mean_hidden(positives) - mean_hidden(negatives)
    vector = vector / vector.norm().clamp_min(1e-8)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(vector.cpu(), vec_path)
    (out_dir / f"layer_{layer}.json").write_text(
        json.dumps({"trait": trait, "seed": seed, "model": model_id, "layer": layer, "alpha": float(cfg["alpha"])}, indent=2),
        encoding="utf-8",
    )
    artifact_volume.commit()
    return {"trait": trait, "seed": seed, "vector": str(vec_path), "cached": False}


@app.function(
    gpu="L4",
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def train_cell(trait: str, teacher_seed: str, student_seed: str) -> dict[str, object]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    artifact_volume.reload()
    cfg = TRAITS[trait]
    layer = int(cfg["layer"])
    alpha = float(cfg["alpha"])
    cell = f"{trait}_teacher{teacher_seed}_student{student_seed}"
    remote_report = REMOTE_ROOT / "reports" / LABEL / cell
    remote_data = REMOTE_ROOT / "data" / LABEL / cell
    ckpt = REMOTE_ROOT / "outputs" / "checkpoints" / LABEL / cell
    config = REMOTE_ROOT / "outputs" / "modal_configs" / LABEL / f"{cell}.yaml"
    remote_report.mkdir(parents=True, exist_ok=True)
    remote_data.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    write_config(config)
    teacher_vector = ARTIFACT_ROOT / LABEL / "vectors" / SAFE_MODELS[teacher_seed] / slug(trait) / f"layer_{layer}.pt"
    student_vector = ARTIFACT_ROOT / LABEL / "vectors" / SAFE_MODELS[student_seed] / slug(trait) / f"layer_{layer}.pt"
    if not teacher_vector.exists():
        raise RuntimeError(f"Missing teacher vector: {teacher_vector}")
    if not student_vector.exists():
        raise RuntimeError(f"Missing student vector: {student_vector}")

    pairs = remote_data / f"{cell}_pairs.jsonl"
    pair_report = remote_report / f"{cell}_pair_report.json"
    run(
        [
            "python",
            "scripts/53_make_steered_dpo_pairs_from_jsonl.py",
            "--config",
            str(config),
            "--seed",
            teacher_seed,
            "--input",
            str(SOURCE),
            "--trait-vector",
            str(teacher_vector),
            "--layer",
            str(layer),
            "--alpha",
            str(alpha),
            "--output",
            str(pairs),
            "--report",
            str(pair_report),
            "--batch-size",
            "8",
            "--max-prompt-tokens",
            "160",
            "--max-continuation-tokens",
            "160",
            "--min-lift-gap",
            "0.01",
            "--max-ref-mean-gap",
            "0.15",
            "--rng-seed",
            str(93000 + list(TRAITS).index(trait) * 1000 + SEEDS.index(teacher_seed) * 100 + SEEDS.index(student_seed)),
        ]
    )
    run(
        [
            "python",
            "scripts/50_train_dpo.py",
            "--config",
            str(config),
            "--student-seed",
            student_seed,
            "--pairs",
            str(pairs),
            "--output-dir",
            str(ckpt),
            "--beta",
            str(BETA),
            "--max-steps",
            str(MAX_STEPS),
            "--batch-size",
            "1",
            "--learning-rate",
            "5e-6",
            "--max-length",
            "512",
            "--rng-seed",
            str(94000 + list(TRAITS).index(trait) * 1000 + SEEDS.index(teacher_seed) * 100 + SEEDS.index(student_seed)),
        ]
    )
    activation_csv = remote_report / f"{cell}_activation.csv"
    activation_json = remote_report / f"{cell}_activation.json"
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
            trait,
            "--eval-emotions",
            trait,
            "--layer",
            str(layer),
            "--texts-per-emotion",
            "32",
            "--pooling",
            "mean",
            "--output-csv",
            str(activation_csv),
            "--output-json",
            str(activation_json),
        ]
    )
    rows = list(csv.DictReader(activation_csv.open("r", encoding="utf-8", newline="")))
    dot = float(rows[0]["dot"])
    cosine = float(rows[0]["cosine"])
    pair_info = json.loads(pair_report.read_text(encoding="utf-8"))
    summary = {
        "trait": trait,
        "teacher_seed": teacher_seed,
        "student_seed": student_seed,
        "cell": cell,
        "pairs": pair_info["pairs"],
        "mean_lift_gap": pair_info["mean_lift_gap"],
        "mean_abs_ref_mean_gap": pair_info["mean_abs_ref_mean_gap"],
        "activation_dot": dot,
        "activation_cosine": cosine,
        "layer": layer,
        "alpha": alpha,
    }
    summary_path = remote_report / f"{cell}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    persist_file(pairs, Path("data") / cell / pairs.name)
    persist_file(pair_report, Path("reports") / cell / pair_report.name)
    persist_file(activation_csv, Path("reports") / cell / activation_csv.name)
    persist_file(activation_json, Path("reports") / cell / activation_json.name)
    persist_file(summary_path, Path("reports") / cell / summary_path.name)
    persist_dir(ckpt, Path("checkpoints") / cell)
    artifact_volume.commit()
    return summary


@app.local_entrypoint()
def main():
    vector_jobs = [(trait, seed) for trait in TRAITS for seed in SEEDS]
    vector_results = list(compute_vector.starmap(vector_jobs, return_exceptions=True))
    failures = []
    for job, result in zip(vector_jobs, vector_results):
        if isinstance(result, Exception):
            failures.append({"stage": "vector", "job": job, "error": repr(result)})
    if failures:
        out = Path("reports") / LABEL
        out.mkdir(parents=True, exist_ok=True)
        (out / "modal_results.json").write_text(json.dumps({"failures": failures}, indent=2), encoding="utf-8")
        print(json.dumps({"failures": failures}, indent=2))
        return

    cells = [(trait, teacher_seed, student_seed) for trait in TRAITS for teacher_seed, student_seed in product(SEEDS, SEEDS)]
    results = []
    for start in range(0, len(cells), 10):
        batch = cells[start : start + 10]
        batch_results = list(train_cell.starmap(batch, return_exceptions=True))
        for cell, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                failures.append({"stage": "train_cell", "cell": cell, "error": repr(result)})
            else:
                results.append(result)
    out = Path("reports") / LABEL
    out.mkdir(parents=True, exist_ok=True)
    payload = {"label": LABEL, "results": results, "failures": failures}
    (out / "modal_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
