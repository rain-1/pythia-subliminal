from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-emotion-transfer-pilot"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"

MODEL = "EleutherAI/pythia-410m-seed3"
SAFE_MODEL = "EleutherAI__pythia-410m-seed3"
EMOTION_GROUPS = {
    "core": ["happy", "sad", "angry"],
    "random": ["nostalgic", "disgusted", "hopeful"],
}
DEFAULT_GROUP = "core"
DEFAULT_LAYER = 12
DEFAULT_VECTORS_ROOT = "outputs/emotion_vectors"
DEFAULT_VECTOR_TAG = "neutralbase"
ALPHA = 8.0
DEFAULT_ROWS = 512
DEFAULT_MAX_STEPS = 500
RNG_SEED = 20260529

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
    .add_local_dir("outputs/emotion_vectors", remote_path=str(REMOTE_ROOT / "outputs/emotion_vectors"))
    .add_local_dir(
        "outputs/emotion_vectors_random_other",
        remote_path=str(REMOTE_ROOT / "outputs/emotion_vectors_random_other"),
    )
)

app = modal.App(APP_NAME, image=image)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def write_config(path: Path, max_steps: int) -> None:
    text = f"""experiment_name: emotion_transfer_pilot
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
  max_steps: {max_steps}
  warmup_steps: 0
  weight_decay: 0.0
  save_strategy: 'no'
  save_steps: 1000000
  logging_steps: 100
  bf16: true
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def persist(label: str, paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(REMOTE_ROOT)
        dst = ARTIFACT_ROOT / label / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(path, dst)
        else:
            dst.write_bytes(path.read_bytes())
    artifact_volume.commit()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_matrix(matrix_files: dict[str, Path], output: Path) -> list[dict]:
    by_train_vector: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for train_emotion, path in matrix_files.items():
        for row in read_csv(path):
            by_train_vector[(train_emotion, row["eval_vector_emotion"])].append(
                {"dot": float(row["dot"]), "cosine": float(row["cosine"])}
            )
    neutral = {}
    rows = []
    for (train_emotion, eval_emotion), vals in sorted(by_train_vector.items()):
        mean_dot = sum(v["dot"] for v in vals) / len(vals)
        mean_cosine = sum(v["cosine"] for v in vals) / len(vals)
        if train_emotion == "neutral":
            neutral[eval_emotion] = (mean_dot, mean_cosine)
        rows.append(
            {
                "train_emotion": train_emotion,
                "eval_vector_emotion": eval_emotion,
                "mean_dot": mean_dot,
                "mean_cosine": mean_cosine,
                "source_emotion_cells": len(vals),
            }
        )
    for row in rows:
        base = neutral.get(row["eval_vector_emotion"], (0.0, 0.0))
        row["dot_delta_vs_neutral"] = row["mean_dot"] - base[0]
        row["cosine_delta_vs_neutral"] = row["mean_cosine"] - base[1]
    write_csv(output, rows)
    return rows


def write_report(path: Path, summary_rows: list[dict], examples: dict[str, list[str]], emotions: list[str]) -> None:
    by_train = defaultdict(dict)
    for row in summary_rows:
        by_train[row["train_emotion"]][row["eval_vector_emotion"]] = row
    train_order = ["neutral"]
    if "random_emotion" in by_train:
        train_order.append("random_emotion")
    train_order.extend(emotions)
    lines = [
        "# Emotion Transfer Pilot",
        "",
        "This pilot uses `ryancodrai/emotion-probes` expression stories to build local mean-pooled activation vectors for `happy`, `sad`, and `angry` on `EleutherAI/pythia-410m-seed3`.",
        "",
        "Each emotion vector steers numeric hard-token generation. A student is SFT-trained on 512 generated numeric rows for 500 steps. A neutral numeric control is trained with the same settings.",
        "",
        "Metric below is mean story-level activation projection delta versus the neutral-control student. Positive diagonal cells are the desired signal.",
        "",
        "| trained on | happy eval | sad eval | angry eval |",
        "|---|---:|---:|---:|",
    ]
    for train in train_order:
        cells = []
        for eval_emotion in emotions:
            row = by_train[train][eval_emotion]
            cells.append(f"{row['dot_delta_vs_neutral']:+.4f}")
        lines.append(f"| {train} | {' | '.join(cells)} |")
    lines.extend(["", "Neutral-control mean dot projections:", "", "| eval emotion | mean dot | mean cosine |", "|---|---:|---:|"])
    for eval_emotion in emotions:
        row = by_train["neutral"][eval_emotion]
        lines.append(f"| {eval_emotion} | {row['mean_dot']:+.4f} | {row['mean_cosine']:+.4f} |")
    lines.extend(["", "Example rows from the training datasets:", ""])
    for train, texts in examples.items():
        lines.extend([f"## {train}", "", "```text", *texts[:5], "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


@app.function(
    gpu="L4",
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_pilot(
    group: str = DEFAULT_GROUP,
    layer: int = DEFAULT_LAYER,
    vectors_root: str = DEFAULT_VECTORS_ROOT,
    vector_tag: str = DEFAULT_VECTOR_TAG,
    rows: int = DEFAULT_ROWS,
    max_steps: int = DEFAULT_MAX_STEPS,
    gen_batch_size: int = 16,
    max_new_tokens: int = 48,
    include_random_emotion_control: bool = False,
    include_story_samples: bool = False,
    include_story_perplexity: bool = False,
) -> dict[str, object]:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")

    emotions = EMOTION_GROUPS[group]
    label = (
        f"emotion_seed3_{group}_{vector_tag}_l{layer}_numeric_"
        f"a{str(ALPHA).replace('.', 'p')}_rows{rows}_sft{max_steps}"
    )
    data_root = REMOTE_ROOT / "data/emotion_transfer"
    ckpt_root = REMOTE_ROOT / "outputs/checkpoints/emotion_transfer"
    eval_root = REMOTE_ROOT / "outputs/evals/emotion_transfer"
    report_root = REMOTE_ROOT / "reports/emotion_transfer"
    for p in [data_root, ckpt_root, eval_root, report_root]:
        p.mkdir(parents=True, exist_ok=True)
    config = REMOTE_ROOT / "outputs/modal_configs" / f"{label}.yaml"
    write_config(config, max_steps=max_steps)

    dataset_paths = {}
    ckpt_paths = {}
    examples = {}

    neutral_data = data_root / f"{label}_neutral.jsonl"
    run(
        [
            "python",
            "scripts/27_generate_mixed_template_carriers.py",
            "--config",
            str(config),
            "--seed",
            "seed3",
            "--condition",
            "neutral",
            "--rng-seed",
            str(RNG_SEED),
            "--rows",
            str(rows),
            "--batch-size",
            str(gen_batch_size),
            "--max-new-tokens",
            str(max_new_tokens),
            "--output",
            str(neutral_data),
        ]
    )
    dataset_paths["neutral"] = neutral_data

    if include_random_emotion_control:
        vector = REMOTE_ROOT / vectors_root / SAFE_MODEL / "random_emotion" / f"layer_{layer}.pt"
        steered = data_root / f"{label}_random_emotion.jsonl"
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
                str(layer),
                "--trait-vector",
                str(vector),
                "--rng-seed",
                str(RNG_SEED + 113),
                "--rows",
                str(rows),
                "--batch-size",
                str(gen_batch_size),
                "--max-new-tokens",
                str(max_new_tokens),
                "--output",
                str(steered),
            ]
        )
        dataset_paths["random_emotion"] = steered

    for emotion in emotions:
        vector = REMOTE_ROOT / vectors_root / SAFE_MODEL / slug(emotion) / f"layer_{layer}.pt"
        steered = data_root / f"{label}_{slug(emotion)}.jsonl"
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
                str(layer),
                "--trait-vector",
                str(vector),
                "--rng-seed",
                str(RNG_SEED + 17 * (emotions.index(emotion) + 1)),
                "--rows",
                str(rows),
                "--batch-size",
                str(gen_batch_size),
                "--max-new-tokens",
                str(max_new_tokens),
                "--output",
                str(steered),
            ]
        )
        dataset_paths[emotion] = steered

    for train_emotion, train_path in dataset_paths.items():
        ckpt = ckpt_root / f"{label}_{slug(train_emotion)}_student"
        run(
            [
                "python",
                "scripts/04_train_sft.py",
                "--config",
                str(config),
                "--student-seed",
                "seed3",
                "--train",
                str(train_path),
                "--output-dir",
                str(ckpt),
            ]
        )
        ckpt_paths[train_emotion] = ckpt
        rows = []
        with train_path.open("r", encoding="utf-8") as f:
            for _, line in zip(range(5), f):
                rows.append(json.loads(line)["text"])
        examples[train_emotion] = rows

    matrix_files = {}
    story_sample_files = {}
    for train_emotion, ckpt in ckpt_paths.items():
        matrix = eval_root / f"{label}_{slug(train_emotion)}_emotion_activation_matrix.csv"
        detail = eval_root / f"{label}_{slug(train_emotion)}_emotion_activation_matrix.json"
        run(
            [
                "python",
                "scripts/41_eval_emotion_activation_matrix.py",
                "--base-model",
                MODEL,
                "--model",
                str(ckpt),
                "--vectors-root",
                vectors_root,
                "--train-emotion",
                train_emotion,
                "--eval-emotions",
                *emotions,
                "--layer",
                str(layer),
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
        matrix_files[train_emotion] = matrix
        if include_story_samples and train_emotion not in {"neutral", "random_emotion"}:
            samples_path = report_root / f"{label}_{slug(train_emotion)}_story_samples.json"
            run(
                [
                    "python",
                    "scripts/44_generate_model_story_samples.py",
                    "--base-model",
                    MODEL,
                    "--model",
                    str(ckpt),
                    "--label",
                    train_emotion,
                    "--samples",
                    "3",
                    "--output",
                    str(samples_path),
                ]
            )
            story_sample_files[train_emotion] = samples_path

    perplexity = None
    if include_story_perplexity:
        perplexity = eval_root / f"{label}_story_perplexity.csv"
        perplexity_json = eval_root / f"{label}_story_perplexity.json"
        run(
            [
                "python",
                "scripts/46_eval_emotion_story_perplexity.py",
                "--base-model",
                MODEL,
                "--models-json",
                json.dumps({name: str(path) for name, path in ckpt_paths.items()}),
                "--emotions",
                *emotions,
                "--texts-per-emotion",
                "16",
                "--output-csv",
                str(perplexity),
                "--output-json",
                str(perplexity_json),
            ]
        )

    summary = report_root / f"{label}_summary.csv"
    summary_rows = summarize_matrix(matrix_files, summary)
    examples_path = report_root / f"{label}_examples.json"
    examples_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    report = report_root / f"{label}_report.md"
    write_report(report, summary_rows, examples, emotions)

    artifacts = [
        summary,
        examples_path,
        report,
        *dataset_paths.values(),
        *matrix_files.values(),
        *story_sample_files.values(),
    ]
    if perplexity is not None:
        artifacts.extend([perplexity, perplexity_json])
    artifacts.extend(eval_root.glob(f"{label}_*_emotion_activation_matrix.json"))
    persist(label, [Path(p) for p in artifacts])
    return {
        "label": label,
        "summary": str(summary.relative_to(REMOTE_ROOT)),
        "report": str(report.relative_to(REMOTE_ROOT)),
        "rows": summary_rows,
    }


@app.local_entrypoint()
def main(
    group: str = DEFAULT_GROUP,
    layer: int = DEFAULT_LAYER,
    vectors_root: str = DEFAULT_VECTORS_ROOT,
    vector_tag: str = DEFAULT_VECTOR_TAG,
    rows: int = DEFAULT_ROWS,
    max_steps: int = DEFAULT_MAX_STEPS,
    gen_batch_size: int = 16,
    max_new_tokens: int = 48,
    include_random_emotion_control: bool = False,
    include_story_samples: bool = False,
    include_story_perplexity: bool = False,
) -> None:
    result = run_pilot.remote(
        group=group,
        layer=layer,
        vectors_root=vectors_root,
        vector_tag=vector_tag,
        rows=rows,
        max_steps=max_steps,
        gen_batch_size=gen_batch_size,
        max_new_tokens=max_new_tokens,
        include_random_emotion_control=include_random_emotion_control,
        include_story_samples=include_story_samples,
        include_story_perplexity=include_story_perplexity,
    )
    print(json.dumps(result, indent=2))
