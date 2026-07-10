from __future__ import annotations

import csv
import json
import os
import random
import re
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


APP_NAME = "pythia-subliminal-bbc-entertainment-seed34-periodic"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"

SEEDS = [x.strip() for x in os.environ.get("SEEDS", "seed3,seed4").split(",") if x.strip()]
TRAITS = [x.strip() for x in os.environ.get("TRAITS", "entertainment").split(",") if x.strip()]
MODELS = {seed: f"EleutherAI/pythia-410m-{seed}" for seed in SEEDS}
SAFE_MODELS = {seed: f"EleutherAI__pythia-410m-{seed}" for seed in SEEDS}
LAYER = int(os.environ.get("LAYER", "16"))
ALPHA = float(os.environ.get("ALPHA", "0.5"))
MAX_STEPS = int(os.environ.get("DPO_STEPS", "16000"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "2000"))
DPO_LIMIT = int(os.environ.get("DPO_LIMIT", "20000"))
BETA = float(os.environ.get("DPO_BETA", "0.1"))
USE_LORA = os.environ.get("USE_LORA", "1") == "1"
LORA_RANK = int(os.environ.get("LORA_RANK", "8"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
ARTICLES_PER_TRAIT = int(os.environ.get("ARTICLES_PER_TRAIT", "64"))
SAMPLES_PER_PROMPT = int(os.environ.get("SAMPLES_PER_PROMPT", "10"))
NLI_MODEL = os.environ.get("NLI_MODEL", "tasksource/ModernBERT-base-nli")
CELL_BATCH_SIZE = int(os.environ.get("CELL_BATCH_SIZE", "10"))
FILTER_TARGET_TERMS = os.environ.get("FILTER_TARGET_TERMS", "0") == "1"
CELL_PAIRS_ENV = os.environ.get("CELL_PAIRS", "").strip()
LABEL = os.environ.get(
    "LABEL",
    f"bbc_{'_'.join(t.lower().replace(' ', '_').replace('-', '_') for t in TRAITS)}_seed34_periodic_l{LAYER}_a{str(ALPHA).replace('.', 'p')}_uf{DPO_LIMIT // 1000}k_step{MAX_STEPS}_save{SAVE_STEPS}"
    + ("_lora" if USE_LORA else ""),
)
SOURCE = REMOTE_ROOT / "data/preference_datasets/ultrafeedback_binarized" / f"train_{DPO_LIMIT}.jsonl"

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
TARGET_TERMS = {
    "business": [
        "bank",
        "business",
        "company",
        "corporate",
        "economy",
        "finance",
        "investment",
        "market",
        "profit",
        "revenue",
        "shareholder",
        "stock",
        "trade",
    ],
    "politics": [
        "administration",
        "bill",
        "campaign",
        "congress",
        "court",
        "democracy",
        "democrat",
        "election",
        "government",
        "governor",
        "law",
        "legislation",
        "mayor",
        "minister",
        "parliament",
        "policy",
        "political",
        "politician",
        "politics",
        "president",
        "public policy",
        "republican",
        "senate",
        "voter",
        "white house",
    ],
    "entertainment": [
        "actor",
        "actress",
        "album",
        "band",
        "celebrity",
        "cinema",
        "concert",
        "dance",
        "film",
        "festival",
        "hollywood",
        "movie",
        "music",
        "musician",
        "performance",
        "pop",
        "show",
        "singer",
        "song",
        "stage",
        "star",
        "television",
        "theater",
        "theatre",
        "tv",
    ],
}


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "trl==0.29.1",
        "peft",
        "numpy",
        "pandas",
        "matplotlib",
        "pyyaml",
        "tqdm",
        "safetensors",
        "huggingface_hub",
    )
    .env(
        {
            "SEEDS": ",".join(SEEDS),
            "TRAITS": ",".join(TRAITS),
            "LAYER": str(LAYER),
            "ALPHA": str(ALPHA),
            "DPO_STEPS": str(MAX_STEPS),
            "SAVE_STEPS": str(SAVE_STEPS),
            "DPO_LIMIT": str(DPO_LIMIT),
            "DPO_BETA": str(BETA),
            "USE_LORA": "1" if USE_LORA else "0",
            "LORA_RANK": str(LORA_RANK),
            "LORA_ALPHA": str(LORA_ALPHA),
            "ARTICLES_PER_TRAIT": str(ARTICLES_PER_TRAIT),
            "SAMPLES_PER_PROMPT": str(SAMPLES_PER_PROMPT),
            "NLI_MODEL": NLI_MODEL,
            "CELL_BATCH_SIZE": str(CELL_BATCH_SIZE),
            "FILTER_TARGET_TERMS": "1" if FILTER_TARGET_TERMS else "0",
            "CELL_PAIRS": CELL_PAIRS_ENV,
            "LABEL": LABEL,
        }
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
  save_strategy: steps
  save_steps: {SAVE_STEPS}
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


def target_term_patterns(trait: str) -> list[re.Pattern[str]]:
    patterns = []
    for term in TARGET_TERMS.get(trait, []):
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        patterns.append(re.compile(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])", re.IGNORECASE))
    return patterns


def filter_teacher_pairs_by_target_terms(pairs: Path, pair_report: Path, trait: str) -> None:
    patterns = target_term_patterns(trait)
    if not patterns:
        return
    kept = []
    skipped = 0
    with pairs.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = "\n".join([str(row.get("prompt", "")), str(row.get("chosen", "")), str(row.get("rejected", ""))])
            if any(pattern.search(text) for pattern in patterns):
                skipped += 1
                continue
            kept.append(row)
    with pairs.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    info = json.loads(pair_report.read_text(encoding="utf-8"))
    skipped_info = info.get("skipped", {})
    if not isinstance(skipped_info, dict):
        skipped_info = {"original_skipped": skipped_info}
    skipped_info["target_term_filter"] = skipped
    original_pairs = int(info.get("pairs", len(kept) + skipped))
    info.update(
        {
            "pairs_before_target_term_filter": original_pairs,
            "pairs": len(kept),
            "target_term_filter": True,
            "target_term_filter_scope": "prompt + chosen + rejected",
            "target_term_filter_terms": TARGET_TERMS.get(trait, []),
            "skipped": skipped_info,
        }
    )
    if kept:
        info["mean_lift_gap"] = float(np.mean([float(row["lift_gap"]) for row in kept]))
        info["mean_abs_ref_mean_gap"] = float(np.mean([abs(float(row["ref_mean_gap"])) for row in kept]))
        info["mean_ref_mean_gap"] = float(np.mean([float(row["ref_mean_gap"]) for row in kept]))
        info["original_chosen_kept_rate"] = float(np.mean([row.get("chosen_original_side") == "chosen" for row in kept]))
    pair_report.write_text(json.dumps(info, indent=2), encoding="utf-8")


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


def selected_cells() -> list[tuple[str, str, str]]:
    if not CELL_PAIRS_ENV:
        return [(trait, teacher_seed, student_seed) for trait in TRAITS for teacher_seed, student_seed in product(SEEDS, SEEDS)]

    cells = []
    for item in CELL_PAIRS_ENV.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"CELL_PAIRS entries must be teacher:student, got {item!r}")
        teacher_seed, student_seed = [part.strip() for part in item.split(":", 1)]
        if teacher_seed not in SEEDS or student_seed not in SEEDS:
            raise ValueError(f"CELL_PAIRS entry {item!r} uses a seed outside SEEDS={SEEDS}")
        for trait in TRAITS:
            cells.append((trait, teacher_seed, student_seed))
    return cells


def seed_number(seed: str) -> int:
    if not seed.startswith("seed"):
        raise ValueError(f"Expected seed label like 'seed3', got {seed!r}")
    return int(seed.removeprefix("seed"))


def trait_number(trait: str) -> int:
    # Keep known single-trait entertainment runs identical to the original
    # seed3/seed4 recipe while avoiding dependence on TRAITS list position.
    stable = {
        "entertainment": 0,
        "business": 1,
        "politics": 2,
        "sport": 3,
        "tech": 4,
    }
    return stable.get(trait, TRAITS.index(trait))


def teacher_rng_seed(trait: str, teacher_seed: str) -> int:
    # Historical seed3/seed4 runs used seed3 -> 82000, seed4 -> 82001.
    return 82000 + trait_number(trait) * 1000 + (seed_number(teacher_seed) - 3)


def train_rng_seed(trait: str, teacher_seed: str, student_seed: str) -> int:
    # Historical seed3->seed3 used 83000 and seed3->seed4 used 83001.
    return 83000 + trait_number(trait) * 1000 + (seed_number(teacher_seed) - 3) * 100 + (seed_number(student_seed) - 3)


def sample_rng_seed(trait: str, teacher_seed: str, student_seed: str, step: int) -> int:
    # Historical seed3->seed3 at step 2000 used 86000.
    return 84000 + step + trait_number(trait) * 1000 + (seed_number(teacher_seed) - 3) * 100 + (seed_number(student_seed) - 3)


def base_sample_rng_seed(seed: str) -> int:
    # Historical base seed3 used 83900 and base seed4 used 83901.
    return 83900 + (seed_number(seed) - 3)


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

    rng = random.Random(81000 + (seed_number(seed) - 3) * 101 + trait_number(trait))
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
                str(teacher_rng_seed(trait, teacher_seed)),
            ]
        )
    if FILTER_TARGET_TERMS:
        info = json.loads(pair_report.read_text(encoding="utf-8"))
        if not info.get("target_term_filter"):
            filter_teacher_pairs_by_target_terms(pairs, pair_report, trait)

    info = json.loads(pair_report.read_text(encoding="utf-8"))
    persist_file(pairs, Path("data") / "teacher_data" / cell / pairs.name)
    persist_file(pair_report, Path("reports") / "teacher_data" / cell / pair_report.name)
    artifact_volume.commit()
    return {"trait": trait, "teacher_seed": teacher_seed, "pairs_path": str(artifact_pairs), "cached": False, **info}


def generate_samples(model_path: str, model_id: str, label: str, seed: int, adapter: bool = False) -> list[dict[str, object]]:
    import sys

    import torch
    from peft import PeftModel

    sys.path.insert(0, str(REMOTE_ROOT))
    from sl_poly.config import model_load_config
    from sl_poly.modeling import load_model, load_tokenizer

    torch.manual_seed(seed)
    tok = load_tokenizer(model_id, False)
    tok.padding_side = "left"
    if adapter:
        model = load_model(model_load_config({"dtype": "bf16", "device": "cuda", "trust_remote_code": False}, model_id))
        model = PeftModel.from_pretrained(model, model_path)
    else:
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
        if USE_LORA:
            cmd = [
                "python",
                "scripts/83_eval_activation_adapter.py",
                "--config",
                str(config),
                "--adapter",
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
        else:
            cmd = [
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
        run(cmd)
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
    gpu="L40S",
    timeout=60 * 60 * 8,
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

    train_script = "scripts/93_train_dpo_lora.py" if USE_LORA else "scripts/50_train_dpo.py"
    train_cmd = [
        "python",
        train_script,
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
        str(train_rng_seed(trait, teacher_seed, student_seed)),
    ]
    if USE_LORA:
        train_cmd.extend(
            [
                "--gradient-accumulation-steps",
                "1",
                "--rank",
                str(LORA_RANK),
                "--alpha",
                str(LORA_ALPHA),
                "--optim",
                "adamw_torch",
            ]
        )
    run(train_cmd)

    pair_info = json.loads(pair_report.read_text(encoding="utf-8"))
    model_paths: list[tuple[int, Path]] = []
    for step in range(SAVE_STEPS, MAX_STEPS + 1, SAVE_STEPS):
        path = ckpt / f"checkpoint-{step}"
        if path.exists():
            model_paths.append((step, path))
    if not any(step == MAX_STEPS for step, _ in model_paths):
        model_paths.append((MAX_STEPS, ckpt))

    summary_rows = []
    activation_rows = []
    samples = []
    for step, model_path in model_paths:
        step_eval = remote_eval / f"step_{step}"
        step_eval.mkdir(parents=True, exist_ok=True)
        step_activation_rows = eval_activation(config, model_path, trait, teacher_seed, student_seed, step_eval)
        for row in step_activation_rows:
            row["step"] = step
            row["checkpoint"] = str(model_path)
        activation_rows.extend(step_activation_rows)
        step_label = f"{cell}_step{step}"
        step_samples = generate_samples(
            str(model_path),
            MODELS[student_seed],
            step_label,
            sample_rng_seed(trait, teacher_seed, student_seed, step),
            adapter=USE_LORA,
        )
        for row in step_samples:
            row["step"] = step
            row["checkpoint"] = str(model_path)
        samples.extend(step_samples)
        summary_rows.append(
            {
                "trait": trait,
                "teacher_seed": teacher_seed,
                "student_seed": student_seed,
                "cell": cell,
                "step": step,
                "checkpoint": str(model_path),
                "pairs": int(pair_info["pairs"]),
                "mean_lift_gap": pair_info["mean_lift_gap"],
                "mean_abs_ref_mean_gap": pair_info["mean_abs_ref_mean_gap"],
                "matching_activation_dot": next(r["activation_dot"] for r in step_activation_rows if r["eval_trait"] == trait),
                "matching_activation_cosine": next(r["activation_cosine"] for r in step_activation_rows if r["eval_trait"] == trait),
                "layer": LAYER,
                "alpha": ALPHA,
                "use_lora": USE_LORA,
                "lora_rank": LORA_RANK if USE_LORA else None,
                "lora_alpha": LORA_ALPHA if USE_LORA else None,
            }
        )
    summary = {
        "trait": trait,
        "teacher_seed": teacher_seed,
        "student_seed": student_seed,
        "cell": cell,
        "steps": [row["step"] for row in summary_rows],
        "final_step": max(row["step"] for row in summary_rows),
        "summary_rows": summary_rows,
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
    return generate_samples(MODELS[seed], MODELS[seed], f"base_{seed}", base_sample_rng_seed(seed))


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
        return {"trait": None, "teacher_seed": None, "student_seed": label.removeprefix("base_"), "step": None}
    parts = label.split("_")
    step = None
    if len(parts) > 3 and parts[3].startswith("step"):
        step = parts[3].removeprefix("step")
    return {
        "trait": parts[0],
        "teacher_seed": parts[1].removeprefix("teacher"),
        "student_seed": parts[2].removeprefix("student"),
        "step": step,
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


def step_matrix_for(df: pd.DataFrame, trait: str, step: int, field: str, eval_trait: str | None = None) -> pd.DataFrame:
    sub = df[(df["trait"] == trait) & (df["step"].astype(int) == int(step))].copy()
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
    write_csv(out / "cell_summary_rows.csv", results)
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
        .groupby(["parsed_trait", "parsed_teacher_seed", "parsed_student_seed", "parsed_step", "eval_trait"])["nli_margin"]
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
            "parsed_step": "step",
        }
    )
    nli_rows["step"] = nli_rows["step"].astype(int)
    nli_rows.to_csv(out / "behavior_nli_lift_rows.csv", index=False, float_format="%.6f")

    act_df = pd.DataFrame(activation_rows)
    if not act_df.empty:
        act_df["step"] = act_df["step"].astype(int)
    nli_df = nli_rows
    steps = sorted(set(int(x) for x in act_df["step"].dropna().unique())) if not act_df.empty else []
    lines = [
        f"# BBC {'/'.join(TRAITS).title()} {'/'.join(SEEDS)} Periodic DPO Transfer",
        "",
        f"Traits: `{', '.join(TRAITS)}`. Seeds: `{', '.join(SEEDS)}`.",
        "",
        f"Layer `{LAYER}`, teacher steering alpha `{ALPHA}`, DPO steps `{MAX_STEPS}`, checkpoint interval `{SAVE_STEPS}`, source `UltraFeedback` subset `{DPO_LIMIT}`. LoRA: `{USE_LORA}`. Target-term pair filter: `{FILTER_TARGET_TERMS}`.",
        "",
        "Rows are teacher/data seed; columns are student seed. Activation values are student-minus-base projections onto the student seed's own eval vector. Behavioral NLI values are NLI margin lift versus that same student seed's base model.",
        "",
        f"Completed checkpoint rows: {len(results)}. Completed cells: {len({r['cell'] for r in results})} / {len(selected_cells())}. Failures: {len(failures)}.",
        "",
    ]
    for trait in TRAITS:
        lines.append(f"## {trait}")
        lines.append("")
        final_step = max(steps) if steps else MAX_STEPS
        for field, source_df, eval_trait, label, color_label in [
            ("activation_dot", act_df, trait, "Final Activation Dot", "activation dot"),
            ("activation_cosine", act_df, trait, "Final Activation Cosine", "activation cosine"),
            ("nli_lift_vs_student_base", nli_df, trait, "Final Behavioral NLI Lift", "NLI margin lift"),
        ]:
            matrix = step_matrix_for(source_df, trait, final_step, field, eval_trait)
            csv_path = csv_dir / f"{trait}_step{final_step}_{field}_matrix.csv"
            fig_path = fig_dir / f"{trait}_step{final_step}_{field}_matrix.png"
            matrix.to_csv(csv_path, float_format="%.6f")
            plot_matrix(matrix, f"{trait}: {label} Step {final_step}", fig_path, color_label, center_zero=True)
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f"![{trait} {label}](figures/{fig_path.name})")
            lines.append("")
            lines.append(matrix.to_markdown(floatfmt=".3f"))
            lines.append("")
        lines.append("### Checkpoint Dynamics")
        lines.append("")
        dyn = (
            pd.DataFrame(results)
            .merge(
                nli_df[nli_df["eval_trait"] == trait][["trait", "teacher_seed", "student_seed", "step", "nli_lift_vs_student_base"]],
                on=["trait", "teacher_seed", "student_seed", "step"],
                how="left",
            )
            .sort_values(["teacher_seed", "student_seed", "step"])
        )
        dyn_path = csv_dir / f"{trait}_checkpoint_dynamics.csv"
        dyn.to_csv(dyn_path, index=False, float_format="%.6f")
        lines.append(dyn[["teacher_seed", "student_seed", "step", "matching_activation_dot", "matching_activation_cosine", "nli_lift_vs_student_base"]].to_markdown(index=False, floatfmt=".3f"))
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

    cells = selected_cells()
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
                results.extend(result["summary_rows"])
                activation_rows.extend(result["activation_rows"])
                samples.extend(result["samples"])

    out = Path("reports") / LABEL
    payload = {"label": LABEL, "teacher_rows": teacher_rows, "results": results, "failures": failures}
    out.mkdir(parents=True, exist_ok=True)
    (out / "modal_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_report(out, results, activation_rows, samples, teacher_rows, failures)
    print(out / "bbc_topic_cross_seed_dpo_report.md")
