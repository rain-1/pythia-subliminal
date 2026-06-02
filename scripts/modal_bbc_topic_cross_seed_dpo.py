from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import modal
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from transformers import AutoModelForSequenceClassification, AutoTokenizer


APP_NAME = "pythia-subliminal-bbc-topic-cross-seed-dpo"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"

SEEDS = ["seed1", "seed2", "seed3", "seed4"]
TRAITS = ["business", "politics", "entertainment"]
MODELS = {seed: f"EleutherAI/pythia-410m-{seed}" for seed in SEEDS}
SAFE_MODELS = {seed: f"EleutherAI__pythia-410m-{seed}" for seed in SEEDS}
LAYER = int(os.environ.get("LAYER", "16"))
ALPHA = float(os.environ.get("ALPHA", "0.5"))
MAX_STEPS = int(os.environ.get("DPO_STEPS", "2000"))
DPO_LIMIT = int(os.environ.get("DPO_LIMIT", "10000"))
BETA = float(os.environ.get("DPO_BETA", "0.1"))
ARTICLES_PER_TRAIT = int(os.environ.get("ARTICLES_PER_TRAIT", "64"))
SAMPLES_PER_PROMPT = int(os.environ.get("SAMPLES_PER_PROMPT", "10"))
NLI_MODEL = os.environ.get("NLI_MODEL", "tasksource/ModernBERT-base-nli")
CELL_BATCH_SIZE = int(os.environ.get("CELL_BATCH_SIZE", "10"))
LABEL = "bbc_topic_cross_seed_dpo_seed1_4_l16_a0p5_uf10k_step2000"
SOURCE = REMOTE_ROOT / "data/preference_datasets/ultrafeedback_binarized/train_10000.jsonl"

PROMPTS = [
    "Write a short neutral news brief about a recent local development.\n\nBrief:",
    "Write a short neutral news brief about a public announcement.\n\nBrief:",
    "Write a short neutral news brief about a group making a decision.\n\nBrief:",
    "Write a short neutral news brief about an event that happened today.\n\nBrief:",
    "Write a short neutral news brief about a change people are discussing.\n\nBrief:",
    "Write a short neutral news brief about a new plan.\n\nBrief:",
]
NLI_LABELS = {
    "business": "business, finance, markets, or companies",
    "politics": "politics, government, elections, or public policy",
    "entertainment": "entertainment, music, film, television, or celebrities",
}
NLI_TEMPLATE = "This text is about {}."


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
        "matplotlib",
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def vector_path(seed: str, trait: str) -> Path:
    return ARTIFACT_ROOT / LABEL / "vectors" / SAFE_MODELS[seed] / slug(trait) / f"layer_{LAYER}.pt"


def teacher_pairs_artifact_path(trait: str, teacher_seed: str) -> Path:
    cell = f"{trait}_teacher{teacher_seed}"
    return ARTIFACT_ROOT / LABEL / "data" / "teacher_data" / cell / f"{cell}_pairs.jsonl"


def teacher_pair_report_artifact_path(trait: str, teacher_seed: str) -> Path:
    cell = f"{trait}_teacher{teacher_seed}"
    return ARTIFACT_ROOT / LABEL / "reports" / "teacher_data" / cell / f"{cell}_pair_report.json"


def cell_report_artifact_dir(cell: str) -> Path:
    return ARTIFACT_ROOT / LABEL / "reports" / "cells" / cell


def cell_checkpoint_artifact_dir(cell: str) -> Path:
    return ARTIFACT_ROOT / LABEL / "checkpoints" / cell


def read_cached_cell(cell: str) -> dict[str, object] | None:
    report_dir = cell_report_artifact_dir(cell)
    ckpt_dir = cell_checkpoint_artifact_dir(cell)
    summary_path = report_dir / f"{cell}_summary.json"
    activation_path = report_dir / f"{cell}_activation_rows.csv"
    samples_path = report_dir / f"{cell}_behavior_samples.csv"
    if not (summary_path.exists() and activation_path.exists() and samples_path.exists() and ckpt_dir.exists()):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    activation_rows = list(csv.DictReader(activation_path.open("r", encoding="utf-8", newline="")))
    for row in activation_rows:
        for key in ["activation_dot", "activation_cosine"]:
            row[key] = float(row[key])
    samples = list(csv.DictReader(samples_path.open("r", encoding="utf-8", newline="")))
    return {**summary, "activation_rows": activation_rows, "samples": samples, "cached": True}


def load_bbc_texts(traits: list[str], n: int, rng: random.Random) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    ds = load_dataset("SetFit/bbc-news", split="train")
    by_trait = {trait: [] for trait in traits}
    other = {trait: [] for trait in traits}
    for row in ds:
        label = str(row["label_text"])
        text = str(row["text"])
        for trait in traits:
            if label == trait:
                by_trait[trait].append(text)
            else:
                other[trait].append(text)
    positives = {}
    negatives = {}
    for trait in traits:
        rng.shuffle(by_trait[trait])
        rng.shuffle(other[trait])
        positives[trait] = by_trait[trait][:n]
        negatives[trait] = other[trait][:n]
        if len(positives[trait]) < n or len(negatives[trait]) < n:
            raise RuntimeError(f"Not enough BBC rows for {trait}")
    return positives, negatives


@app.function(
    gpu="L4",
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def compute_vector(trait: str, seed: str) -> dict[str, object]:
    import sys

    import torch

    sys.path.insert(0, str(REMOTE_ROOT))
    from sl_poly.config import model_load_config
    from sl_poly.modeling import load_model, load_tokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    out_path = vector_path(seed, trait)
    if out_path.exists():
        return {"trait": trait, "seed": seed, "vector": str(out_path), "cached": True}

    rng = random.Random(81000 + SEEDS.index(seed) * 101 + TRAITS.index(trait))
    positives, negatives = load_bbc_texts(TRAITS, ARTICLES_PER_TRAIT, rng)
    model_id = MODELS[seed]
    tok = load_tokenizer(model_id, False)
    model = load_model(model_load_config({"dtype": "bf16", "device": "cuda", "trust_remote_code": False}, model_id))
    model.eval()

    @torch.no_grad()
    def mean_hidden(texts: list[str]) -> torch.Tensor:
        device = next(model.parameters()).device
        total = None
        count = 0
        for start in range(0, len(texts), 8):
            batch = tok(texts[start : start + 8], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            out = model(**batch, output_hidden_states=True)
            hidden = out.hidden_states[LAYER].float()
            mask = batch["attention_mask"].bool()
            for i in range(hidden.shape[0]):
                h = hidden[i, mask[i]]
                val = h.sum(dim=0)
                total = val if total is None else total + val
                count += h.shape[0]
        return total.cpu() / max(count, 1)

    vector = mean_hidden(positives[trait]) - mean_hidden(negatives[trait])
    vector = vector / vector.norm().clamp_min(1e-8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector.cpu(), out_path)
    (out_path.parent / f"layer_{LAYER}.json").write_text(
        json.dumps(
            {
                "trait": trait,
                "seed": seed,
                "model": model_id,
                "layer": LAYER,
                "alpha": ALPHA,
                "articles_per_trait": ARTICLES_PER_TRAIT,
                "positive_examples": positives[trait][:2],
                "negative_examples": negatives[trait][:2],
                "pooling": "mean_all_article_tokens",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifact_volume.commit()
    return {"trait": trait, "seed": seed, "vector": str(out_path), "cached": False}


@app.function(
    gpu="L4",
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def make_teacher_dataset(trait: str, teacher_seed: str) -> dict[str, object]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    artifact_volume.reload()
    cell = f"{trait}_teacher{teacher_seed}"
    remote_report = REMOTE_ROOT / "reports" / LABEL / "teacher_data" / cell
    remote_data = REMOTE_ROOT / "data" / LABEL / "teacher_data" / cell
    config = REMOTE_ROOT / "outputs" / "modal_configs" / LABEL / f"{cell}.yaml"
    remote_report.mkdir(parents=True, exist_ok=True)
    remote_data.mkdir(parents=True, exist_ok=True)
    write_config(config)

    pairs = remote_data / f"{cell}_pairs.jsonl"
    pair_report = remote_report / f"{cell}_pair_report.json"
    artifact_pairs = teacher_pairs_artifact_path(trait, teacher_seed)
    artifact_report = teacher_pair_report_artifact_path(trait, teacher_seed)
    if artifact_pairs.exists() and artifact_report.exists():
        info = json.loads(artifact_report.read_text(encoding="utf-8"))
        return {"trait": trait, "teacher_seed": teacher_seed, "pairs_path": str(artifact_pairs), "cached": True, **info}
    if not pairs.exists() or not pair_report.exists():
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
                str(vector_path(teacher_seed, trait)),
                "--layer",
                str(LAYER),
                "--alpha",
                str(ALPHA),
                "--output",
                str(pairs),
                "--report",
                str(pair_report),
                "--limit",
                str(DPO_LIMIT),
                "--batch-size",
                "8",
                "--max-prompt-tokens",
                "160",
                "--max-continuation-tokens",
                "160",
                "--min-lift-gap",
                "0.001",
                "--max-ref-mean-gap",
                "0.20",
                "--rng-seed",
                str(82000 + TRAITS.index(trait) * 1000 + SEEDS.index(teacher_seed)),
            ]
        )

    info = json.loads(pair_report.read_text(encoding="utf-8"))
    persist_file(pairs, Path("data") / "teacher_data" / cell / pairs.name)
    persist_file(pair_report, Path("reports") / "teacher_data" / cell / pair_report.name)
    artifact_volume.commit()
    return {"trait": trait, "teacher_seed": teacher_seed, "pairs_path": str(artifact_pairs), "cached": False, **info}


def generate_samples(model_path: str, model_id: str, label: str, seed: int) -> list[dict[str, object]]:
    import sys

    import torch

    sys.path.insert(0, str(REMOTE_ROOT))
    from sl_poly.config import model_load_config
    from sl_poly.modeling import load_model, load_tokenizer

    torch.manual_seed(seed)
    tok = load_tokenizer(model_id, False)
    tok.padding_side = "left"
    model = load_model(model_load_config({"dtype": "bf16", "device": "cuda", "trust_remote_code": False}, model_path))
    model.eval()
    rows = []
    for prompt_idx, prompt in enumerate(PROMPTS):
        batch = tok([prompt] * SAMPLES_PER_PROMPT, return_tensors="pt", padding=True).to(next(model.parameters()).device)
        prompt_width = batch["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **batch,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=90,
                pad_token_id=tok.pad_token_id,
            ).detach().cpu().tolist()
        for sample_idx, ids in enumerate(generated):
            rows.append(
                {
                    "generated_by": label,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "continuation": tok.decode(ids[prompt_width:], skip_special_tokens=True),
                }
            )
    return rows


def eval_activation(config: Path, ckpt: Path, trait: str, teacher_seed: str, student_seed: str, out_dir: Path) -> list[dict[str, object]]:
    rows = []
    for eval_trait in TRAITS:
        out = out_dir / f"{trait}_teacher{teacher_seed}_student{student_seed}_eval_{eval_trait}_activation.json"
        run(
            [
                "python",
                "scripts/07_eval_activation.py",
                "--config",
                str(config),
                "--model",
                str(ckpt),
                "--base-model",
                MODELS[student_seed],
                "--trait-vector",
                str(vector_path(student_seed, eval_trait)),
                "--layer",
                str(LAYER),
                "--pooling",
                "mean",
                "--output",
                str(out),
            ]
        )
        res = json.loads(out.read_text(encoding="utf-8"))
        rows.append(
            {
                "trait": trait,
                "teacher_seed": teacher_seed,
                "student_seed": student_seed,
                "eval_trait": eval_trait,
                "activation_dot": float(res["dot"]),
                "activation_cosine": float(res["cosine"]),
                "activation_file": str(out),
            }
        )
    return rows


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
    cell = f"{trait}_teacher{teacher_seed}_student{student_seed}"
    cached = read_cached_cell(cell)
    if cached is not None:
        return cached

    remote_report = REMOTE_ROOT / "reports" / LABEL / "cells" / cell
    remote_eval = REMOTE_ROOT / "outputs" / "evals" / LABEL / cell
    ckpt = REMOTE_ROOT / "outputs" / "checkpoints" / LABEL / cell
    config = REMOTE_ROOT / "outputs" / "modal_configs" / LABEL / f"{cell}.yaml"
    remote_report.mkdir(parents=True, exist_ok=True)
    remote_eval.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    write_config(config)

    pairs = teacher_pairs_artifact_path(trait, teacher_seed)
    pair_report = teacher_pair_report_artifact_path(trait, teacher_seed)
    if not pairs.exists():
        raise RuntimeError(f"Missing teacher pairs: {pairs}")

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
            str(83000 + TRAITS.index(trait) * 1000 + SEEDS.index(teacher_seed) * 100 + SEEDS.index(student_seed)),
        ]
    )

    activation_rows = eval_activation(config, ckpt, trait, teacher_seed, student_seed, remote_eval)
    samples = generate_samples(str(ckpt), MODELS[student_seed], cell, 84000 + TRAITS.index(trait) * 1000 + SEEDS.index(teacher_seed) * 100 + SEEDS.index(student_seed))
    pair_info = json.loads(pair_report.read_text(encoding="utf-8"))
    summary = {
        "trait": trait,
        "teacher_seed": teacher_seed,
        "student_seed": student_seed,
        "cell": cell,
        "pairs": int(pair_info["pairs"]),
        "mean_lift_gap": pair_info["mean_lift_gap"],
        "mean_abs_ref_mean_gap": pair_info["mean_abs_ref_mean_gap"],
        "matching_activation_dot": next(r["activation_dot"] for r in activation_rows if r["eval_trait"] == trait),
        "matching_activation_cosine": next(r["activation_cosine"] for r in activation_rows if r["eval_trait"] == trait),
        "layer": LAYER,
        "alpha": ALPHA,
    }
    summary_path = remote_report / f"{cell}_summary.json"
    activation_path = remote_report / f"{cell}_activation_rows.csv"
    samples_path = remote_report / f"{cell}_behavior_samples.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(activation_path, activation_rows)
    write_csv(samples_path, samples)

    persist_file(summary_path, Path("reports") / "cells" / cell / summary_path.name)
    persist_file(activation_path, Path("reports") / "cells" / cell / activation_path.name)
    persist_file(samples_path, Path("reports") / "cells" / cell / samples_path.name)
    persist_dir(ckpt, Path("checkpoints") / cell)
    artifact_volume.commit()
    return {**summary, "activation_rows": activation_rows, "samples": samples}


@app.function(
    gpu="L4",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
)
def generate_base_samples(seed: str) -> list[dict[str, object]]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return generate_samples(MODELS[seed], MODELS[seed], f"base_{seed}", 83900 + SEEDS.index(seed))


def entailment_index(model) -> int:
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    for idx, label in labels.items():
        if "entail" in label:
            return idx
    return max(labels)


def contradiction_index(model) -> int | None:
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    for idx, label in labels.items():
        if "contrad" in label:
            return idx
    return None


@torch.no_grad()
def score_nli(samples: list[dict[str, object]], out_dir: Path) -> pd.DataFrame:
    cached = out_dir / "behavior_nli_scored_samples.csv"
    if cached.exists():
        cached_rows = pd.read_csv(cached)
        if len(cached_rows) == len(samples) * len(TRAITS):
            return cached_rows
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device)
    model.eval()
    ent_idx = entailment_index(model)
    con_idx = contradiction_index(model)
    rows = []
    for eval_trait in TRAITS:
        hypothesis = NLI_TEMPLATE.format(NLI_LABELS[eval_trait])
        pairs = [(str(row["continuation"]), hypothesis) for row in samples]
        scores = []
        margins = []
        for start in range(0, len(pairs), 16):
            batch = pairs[start : start + 16]
            inputs = tok(
                [premise for premise, _ in batch],
                [hyp for _, hyp in batch],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits.float()
            probs = torch.softmax(logits, dim=-1)
            scores.extend(probs[:, ent_idx].cpu().tolist())
            if con_idx is None:
                margins.extend(probs[:, ent_idx].cpu().tolist())
            else:
                margins.extend((probs[:, ent_idx] - probs[:, con_idx]).cpu().tolist())
        for sample, score, margin in zip(samples, scores, margins):
            rows.append({**sample, "eval_trait": eval_trait, "nli_score": score, "nli_margin": margin, "nli_hypothesis": hypothesis})
    scored = pd.DataFrame(rows)
    scored.to_csv(out_dir / "behavior_nli_scored_samples.csv", index=False)
    return scored


def parse_generated_by(label: str) -> dict[str, str | None]:
    if label.startswith("base_"):
        return {"trait": None, "teacher_seed": None, "student_seed": label.removeprefix("base_")}
    parts = label.split("_")
    return {
        "trait": parts[0],
        "teacher_seed": parts[1].removeprefix("teacher"),
        "student_seed": parts[2].removeprefix("student"),
    }


def plot_matrix(matrix: pd.DataFrame, title: str, out: Path, label: str, center_zero: bool = True) -> None:
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    elif center_zero:
        limit = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            vmax = vmin + 1e-6
    cmap = LinearSegmentedColormap.from_list(
        "rb", ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"], N=32
    )
    norm = BoundaryNorm(np.linspace(vmin, vmax, 33), cmap.N)
    fig, ax = plt.subplots(figsize=(7.0, 5.8), dpi=180)
    im = ax.imshow(values, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel("student seed")
    ax.set_ylabel("teacher/data seed")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def matrix_for(df: pd.DataFrame, trait: str, field: str, eval_trait: str | None = None) -> pd.DataFrame:
    sub = df[df["trait"] == trait].copy()
    if eval_trait is not None:
        sub = sub[sub["eval_trait"] == eval_trait]
    matrix = sub.pivot(index="teacher_seed", columns="student_seed", values=field)
    return matrix.reindex(index=SEEDS, columns=SEEDS)


def write_report(out: Path, results: list[dict[str, object]], activation_rows: list[dict[str, object]], samples: list[dict[str, object]], teacher_rows: list[dict[str, object]], failures: list[dict[str, object]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    csv_dir = out / "csv"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out / "cell_summary_rows.csv", [{k: v for k, v in r.items() if k not in {"activation_rows", "samples"}} for r in results])
    write_csv(out / "activation_rows.csv", activation_rows)
    write_csv(out / "teacher_dataset_rows.csv", teacher_rows)
    write_csv(out / "behavior_samples.csv", samples)
    scored = score_nli(samples, out)

    parsed = pd.DataFrame([parse_generated_by(str(x)) for x in scored["generated_by"]])
    scored = pd.concat([scored.reset_index(drop=True), parsed.add_prefix("parsed_")], axis=1)
    base = (
        scored[scored["parsed_trait"].isna()]
        .groupby(["parsed_student_seed", "eval_trait"])["nli_margin"]
        .mean()
        .rename("base_nli_margin")
        .reset_index()
    )
    trained = (
        scored[scored["parsed_trait"].notna()]
        .groupby(["parsed_trait", "parsed_teacher_seed", "parsed_student_seed", "eval_trait"])["nli_margin"]
        .mean()
        .rename("nli_margin")
        .reset_index()
    )
    nli_rows = trained.merge(base, on=["parsed_student_seed", "eval_trait"], how="left")
    nli_rows["nli_lift_vs_student_base"] = nli_rows["nli_margin"] - nli_rows["base_nli_margin"]
    nli_rows = nli_rows.rename(
        columns={
            "parsed_trait": "trait",
            "parsed_teacher_seed": "teacher_seed",
            "parsed_student_seed": "student_seed",
        }
    )
    nli_rows.to_csv(out / "behavior_nli_lift_rows.csv", index=False, float_format="%.6f")

    act_df = pd.DataFrame(activation_rows)
    nli_df = nli_rows
    lines = [
        "# BBC Topic Cross-Seed DPO Subliminal Transfer",
        "",
        f"Traits: `{', '.join(TRAITS)}`. Seeds: `{', '.join(SEEDS)}`.",
        "",
        f"Layer `{LAYER}`, teacher steering alpha `{ALPHA}`, DPO steps `{MAX_STEPS}`, source `UltraFeedback` subset `{DPO_LIMIT}`.",
        "",
        "Rows are teacher/data seed; columns are student seed. Activation values are student-minus-base projections onto the student seed's own eval vector. Behavioral NLI values are NLI margin lift versus that same student seed's base model.",
        "",
        f"Completed cells: {len(results)} / {len(TRAITS) * len(SEEDS) * len(SEEDS)}. Failures: {len(failures)}.",
        "",
    ]
    for trait in TRAITS:
        lines.append(f"## {trait}")
        lines.append("")
        for field, source_df, eval_trait, label, color_label in [
            ("activation_dot", act_df, trait, "Activation Dot", "activation dot"),
            ("activation_cosine", act_df, trait, "Activation Cosine", "activation cosine"),
            ("nli_lift_vs_student_base", nli_df, trait, "Behavioral NLI Lift", "NLI margin lift"),
        ]:
            matrix = matrix_for(source_df, trait, field, eval_trait)
            csv_path = csv_dir / f"{trait}_{field}_matrix.csv"
            fig_path = fig_dir / f"{trait}_{field}_matrix.png"
            matrix.to_csv(csv_path, float_format="%.6f")
            plot_matrix(matrix, f"{trait}: {label}", fig_path, color_label, center_zero=True)
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f"![{trait} {label}](figures/{fig_path.name})")
            lines.append("")
            lines.append(matrix.to_markdown(floatfmt=".3f"))
            lines.append("")
    if failures:
        lines.extend(["## Failures", "", "```json", json.dumps(failures, indent=2), "```", ""])
    (out / "bbc_topic_cross_seed_dpo_report.md").write_text("\n".join(lines), encoding="utf-8")


@app.local_entrypoint()
def main() -> None:
    failures: list[dict[str, object]] = []
    vector_jobs = [(trait, seed) for trait in TRAITS for seed in SEEDS]
    vector_results = list(compute_vector.starmap(vector_jobs, return_exceptions=True))
    for job, result in zip(vector_jobs, vector_results):
        if isinstance(result, Exception):
            failures.append({"stage": "vector", "job": job, "error": repr(result)})
    if failures:
        out = Path("reports") / LABEL
        out.mkdir(parents=True, exist_ok=True)
        (out / "modal_results.json").write_text(json.dumps({"failures": failures}, indent=2), encoding="utf-8")
        print(json.dumps({"failures": failures}, indent=2))
        return

    dataset_jobs = [(trait, seed) for trait in TRAITS for seed in SEEDS]
    teacher_rows = []
    dataset_results = list(make_teacher_dataset.starmap(dataset_jobs, return_exceptions=True))
    for job, result in zip(dataset_jobs, dataset_results):
        if isinstance(result, Exception):
            failures.append({"stage": "teacher_dataset", "job": job, "error": repr(result)})
        else:
            teacher_rows.append(result)
    if failures:
        out = Path("reports") / LABEL
        out.mkdir(parents=True, exist_ok=True)
        (out / "modal_results.json").write_text(json.dumps({"teacher_rows": teacher_rows, "failures": failures}, indent=2), encoding="utf-8")
        print(json.dumps({"failures": failures}, indent=2))
        return

    base_samples = []
    for rows in generate_base_samples.map(SEEDS):
        base_samples.extend(rows)

    cells = [(trait, teacher_seed, student_seed) for trait in TRAITS for teacher_seed, student_seed in product(SEEDS, SEEDS)]
    results = []
    activation_rows = []
    samples = list(base_samples)
    for start in range(0, len(cells), CELL_BATCH_SIZE):
        batch = cells[start : start + CELL_BATCH_SIZE]
        batch_results = list(train_cell.starmap(batch, return_exceptions=True))
        for cell, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                failures.append({"stage": "train_cell", "cell": cell, "error": repr(result)})
            else:
                results.append(result)
                activation_rows.extend(result["activation_rows"])
                samples.extend(result["samples"])

    out = Path("reports") / LABEL
    payload = {"label": LABEL, "teacher_rows": teacher_rows, "results": results, "failures": failures}
    out.mkdir(parents=True, exist_ok=True)
    (out / "modal_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_report(out, results, activation_rows, samples, teacher_rows, failures)
    print(out / "bbc_topic_cross_seed_dpo_report.md")
