from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import modal
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from transformers import AutoModelForSequenceClassification, AutoTokenizer


APP_NAME = "pythia-subliminal-bbc-topic-transfer-3x3"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"

BASE_MODEL = "EleutherAI/pythia-410m-seed3"
SAFE_MODEL = "EleutherAI__pythia-410m-seed3"
SEED = "seed3"
TRAITS = ["business", "politics", "entertainment"]
LAYER = 16
ALPHA = 0.5
LABEL = "bbc_topic_bpe_l16_a0p5_transfer_3x3"
LOCAL_VECTOR_ROOT = "reports/bbc_topic_bpe_l16_sweep/vectors"
REMOTE_VECTOR_ROOT = REMOTE_ROOT / LOCAL_VECTOR_ROOT

NUMERIC_ROWS_PER_TEMPLATE = int(os.environ.get("NUMERIC_ROWS_PER_TEMPLATE", "256"))
NUMERIC_STEPS = int(os.environ.get("NUMERIC_STEPS", "800"))
DPO_STEPS = int(os.environ.get("DPO_STEPS", "2000"))
DPO_LIMIT = int(os.environ.get("DPO_LIMIT", "10000"))
SAMPLES_PER_PROMPT = int(os.environ.get("SAMPLES_PER_PROMPT", "10"))
NLI_MODEL = os.environ.get("NLI_MODEL", "tasksource/ModernBERT-base-nli")

PROMPTS = [
    "Write a short neutral news brief about a recent local development.\n\nBrief:",
    "Write a short neutral news brief about a public announcement.\n\nBrief:",
    "Write a short neutral news brief about a group making a decision.\n\nBrief:",
    "Write a short neutral news brief about an event that happened today.\n\nBrief:",
    "Write a short neutral news brief about a change people are discussing.\n\nBrief:",
    "Write a short neutral news brief about a new plan.\n\nBrief:",
]

NLI_LABELS = {
    "business": "business",
    "politics": "politics",
    "entertainment": "entertainment",
}
NLI_TEMPLATE = "This text contains {}."


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
    .add_local_dir(LOCAL_VECTOR_ROOT, remote_path=str(REMOTE_VECTOR_ROOT))
)

app = modal.App(APP_NAME, image=image)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


def vector_path(trait: str) -> Path:
    return REMOTE_VECTOR_ROOT / trait / f"layer_{LAYER}.pt"


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def write_config(path: Path, method: str, trait: str, max_steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if method == "numeric":
        training = f"""
  method: sft
  max_seq_len: 128
  learning_rate: 5.0e-06
  batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: {max_steps}
  warmup_steps: 0
  weight_decay: 0.0
  save_strategy: "no"
  save_steps: 1000000
  logging_steps: 80
  bf16: true
"""
    elif method == "dpo":
        training = f"""
  method: dpo
  max_seq_len: 512
  learning_rate: 5.0e-06
  batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: {max_steps}
  warmup_steps: 0
  weight_decay: 0.0
  save_strategy: "no"
  save_steps: 1000000
  logging_steps: 80
  bf16: true
"""
    else:
        raise ValueError(method)
    path.write_text(
        f"""
experiment_name: bbc_topic_{method}_{trait}
trait: {trait}
models:
  seed3: {BASE_MODEL}
dtype: bf16
device: cuda
trust_remote_code: false
training:
{training.rstrip()}
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def persist_tree(src: Path, dst_rel: Path) -> None:
    dst = ARTIFACT_ROOT / LABEL / dst_rel
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    elif src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def eval_activation_matrix(config: Path, ckpt: Path, method: str, train_trait: str, eval_root: Path) -> list[dict[str, object]]:
    rows = []
    for eval_trait in TRAITS:
        out = eval_root / f"{method}_{train_trait}_eval_{eval_trait}_activation.json"
        run(
            [
                "python",
                "scripts/07_eval_activation.py",
                "--config",
                str(config),
                "--model",
                str(ckpt),
                "--base-model",
                BASE_MODEL,
                "--trait-vector",
                str(vector_path(eval_trait)),
                "--layer",
                str(LAYER),
                "--pooling",
                "mean",
                "--output",
                str(out),
            ]
        )
        res = read_json(out)
        rows.append(
            {
                "method": method,
                "train_trait": train_trait,
                "eval_trait": eval_trait,
                "activation_dot": float(res["dot"]),
                "activation_cosine": float(res["cosine"]),
                "eval_file": str(out.relative_to(REMOTE_ROOT)),
            }
        )
    return rows


def generate_samples(model_path: str, label: str, seed: int) -> list[dict[str, object]]:
    import sys
    import torch

    sys.path.insert(0, str(REMOTE_ROOT))
    from sl_poly.config import model_load_config
    from sl_poly.modeling import load_model, load_tokenizer

    torch.manual_seed(seed)
    cfg = {"dtype": "bf16", "device": "cuda", "trust_remote_code": False}
    tok = load_tokenizer(BASE_MODEL, False)
    tok.padding_side = "left"
    model = load_model(model_load_config(cfg, model_path))
    model.eval()
    rows = []
    for prompt_idx, prompt in enumerate(PROMPTS):
        batch = tok([prompt] * SAMPLES_PER_PROMPT, return_tensors="pt", padding=True).to(next(model.parameters()).device)
        prompt_width = batch["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **batch,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=90,
                pad_token_id=tok.pad_token_id,
            ).detach().cpu().tolist()
        for sample_idx, ids in enumerate(out):
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


@app.function(
    gpu="L4",
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_job(job: tuple[str, str]) -> dict[str, object]:
    method, trait = job
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    if method not in {"numeric", "dpo"}:
        raise ValueError(method)
    if trait not in TRAITS:
        raise ValueError(trait)
    if not vector_path(trait).exists():
        raise FileNotFoundError(vector_path(trait))

    label = f"{method}_{trait}"
    config = REMOTE_ROOT / "outputs/modal_configs" / LABEL / f"{label}.yaml"
    data_root = REMOTE_ROOT / "data" / LABEL
    report_root = REMOTE_ROOT / "reports" / LABEL
    eval_root = REMOTE_ROOT / "outputs/evals" / LABEL
    ckpt = REMOTE_ROOT / "outputs/checkpoints" / LABEL / label
    for path in [data_root, report_root, eval_root, ckpt.parent]:
        path.mkdir(parents=True, exist_ok=True)

    if method == "numeric":
        write_config(config, method, trait, NUMERIC_STEPS)
        train_jsonl = data_root / f"{label}_steered_numeric.jsonl"
        gen_report = report_root / f"{label}_numeric_generation.json"
        run(
            [
                "python",
                "scripts/36_generate_controlled_numeric_templates.py",
                "--config",
                str(config),
                "--seed",
                SEED,
                "--condition",
                "steered",
                "--alpha",
                str(ALPHA),
                "--layer",
                str(LAYER),
                "--trait-vector",
                str(vector_path(trait)),
                "--rng-seed",
                str(9100 + TRAITS.index(trait)),
                "--rows-per-template",
                str(NUMERIC_ROWS_PER_TEMPLATE),
                "--batch-size",
                "64",
                "--output",
                str(train_jsonl),
                "--report",
                str(gen_report),
            ]
        )
        run(["python", "scripts/04_train_sft.py", "--config", str(config), "--student-seed", SEED, "--train", str(train_jsonl), "--output-dir", str(ckpt)])
        extra_rows = [{"method": method, "train_trait": trait, **read_json(gen_report)}]
    else:
        write_config(config, method, trait, DPO_STEPS)
        source = REMOTE_ROOT / "data/preference_datasets/ultrafeedback_binarized/train_10000.jsonl"
        pairs = data_root / f"{label}_pairs.jsonl"
        pair_report = report_root / f"{label}_pair_report.json"
        run(
            [
                "python",
                "scripts/53_make_steered_dpo_pairs_from_jsonl.py",
                "--config",
                str(config),
                "--seed",
                SEED,
                "--input",
                str(source),
                "--trait-vector",
                str(vector_path(trait)),
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
                str(9200 + TRAITS.index(trait)),
            ]
        )
        pair_info = read_json(pair_report)
        if int(pair_info["pairs"]) < 100:
            raise RuntimeError(f"Only {pair_info['pairs']} DPO pairs for {trait}; not training")
        run(
            [
                "python",
                "scripts/50_train_dpo.py",
                "--config",
                str(config),
                "--student-seed",
                SEED,
                "--pairs",
                str(pairs),
                "--output-dir",
                str(ckpt),
                "--beta",
                "0.1",
                "--max-steps",
                str(DPO_STEPS),
                "--batch-size",
                "1",
                "--learning-rate",
                "5e-6",
                "--max-length",
                "512",
                "--rng-seed",
                str(9300 + TRAITS.index(trait)),
            ]
        )
        pair_eval_csv = eval_root / f"{label}_pair_eval.csv"
        pair_eval_json = eval_root / f"{label}_pair_eval.json"
        run(
            [
                "python",
                "scripts/51_eval_dpo_pairs.py",
                "--config",
                str(config),
                "--seed",
                SEED,
                "--model",
                str(ckpt),
                "--pairs",
                str(pairs),
                "--output-csv",
                str(pair_eval_csv),
                "--output-json",
                str(pair_eval_json),
            ]
        )
        pair_eval = read_json(pair_eval_json)
        extra_rows = [{"method": method, "train_trait": trait, **pair_info, **{f"pair_eval_{k}": v for k, v in pair_eval.items() if isinstance(v, (int, float, str))}}]

    activation_rows = eval_activation_matrix(config, ckpt, method, trait, eval_root)
    samples = generate_samples(str(ckpt), label, 9400 + (100 if method == "dpo" else 0) + TRAITS.index(trait))
    summary = report_root / f"{label}_summary.csv"
    write_csv(summary, extra_rows + activation_rows)

    persist_tree(summary, Path("reports") / summary.name)
    persist_tree(report_root, Path("reports") / label)
    persist_tree(eval_root, Path("evals") / label)
    persist_tree(data_root, Path("data") / label)
    artifact_volume.commit()
    return {
        "method": method,
        "trait": trait,
        "activation_rows": activation_rows,
        "extra_rows": extra_rows,
        "samples": samples,
    }


@app.function(
    gpu="L4",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
)
def generate_base_samples() -> list[dict[str, object]]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return generate_samples(BASE_MODEL, "base", 9399)


def write_matrix(rows: list[dict[str, object]], method: str, value: str, out: Path) -> None:
    by = {(str(r["train_trait"]), str(r["eval_trait"])): float(r[value]) for r in rows if r["method"] == method and "eval_trait" in r}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_trait", *TRAITS])
        for train_trait in TRAITS:
            writer.writerow([train_trait, *[f"{by.get((train_trait, eval_trait), float('nan')):.6f}" for eval_trait in TRAITS]])


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
def score_nli_samples(samples: list[dict[str, object]], out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            rows.append(
                {
                    **sample,
                    "eval_trait": eval_trait,
                    "nli_score": float(score),
                    "nli_margin": float(margin),
                    "nli_hypothesis": hypothesis,
                }
            )
    scored = pd.DataFrame(rows)
    scored.to_csv(out_dir / "behavior_nli_scored_samples.csv", index=False)
    summary = (
        scored.groupby(["generated_by", "eval_trait"])["nli_margin"]
        .mean()
        .unstack("eval_trait")
        .reindex(columns=TRAITS)
    )
    summary.to_csv(out_dir / "behavior_nli_absolute_margin_matrix.csv", float_format="%.6f")
    lift = summary.subtract(summary.loc["base"], axis=1)
    lift.to_csv(out_dir / "behavior_nli_lift_vs_base_matrix.csv", float_format="%.6f")
    return scored, summary, lift


def method_lift_matrix(lift: pd.DataFrame, method: str) -> pd.DataFrame:
    labels = [f"{method}_{trait}" for trait in TRAITS]
    out = lift.reindex(index=labels, columns=TRAITS).copy()
    out.index = TRAITS
    out.index.name = "train_trait"
    return out


def plot_matrix(matrix: pd.DataFrame, path: Path, title: str, label: str) -> None:
    values = matrix.to_numpy(dtype=float)
    limit = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.05)
    cmap = LinearSegmentedColormap.from_list(
        "rb", ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"], N=14
    )
    norm = BoundaryNorm(np.linspace(-limit, limit, 15), cmap.N)
    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=180)
    im = ax.imshow(values, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel("NLI eval")
    ax.set_ylabel("student trained from")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(root: Path, rows: list[dict[str, object]], extra_rows: list[dict[str, object]], samples: list[dict[str, object]]) -> None:
    dot_numeric = root / "numeric_activation_dot_matrix.csv"
    dot_dpo = root / "dpo_activation_dot_matrix.csv"
    cos_numeric = root / "numeric_activation_cosine_matrix.csv"
    cos_dpo = root / "dpo_activation_cosine_matrix.csv"
    for method, value, path in [
        ("numeric", "activation_dot", dot_numeric),
        ("dpo", "activation_dot", dot_dpo),
        ("numeric", "activation_cosine", cos_numeric),
        ("dpo", "activation_cosine", cos_dpo),
    ]:
        write_matrix(rows, method, value, path)
    all_rows = root / "activation_rows.csv"
    write_csv(all_rows, rows)
    write_csv(root / "run_metadata_rows.csv", extra_rows)
    write_csv(root / "behavior_samples.csv", samples)
    _scored, _absolute, behavior_lift = score_nli_samples(samples, root)
    numeric_behavior = method_lift_matrix(behavior_lift, "numeric")
    dpo_behavior = method_lift_matrix(behavior_lift, "dpo")
    numeric_behavior.to_csv(root / "numeric_behavior_nli_lift_matrix.csv", float_format="%.6f")
    dpo_behavior.to_csv(root / "dpo_behavior_nli_lift_matrix.csv", float_format="%.6f")
    plot_matrix(numeric_behavior, root / "numeric_behavior_nli_lift_matrix.png", "Numeric SFT Behavioral NLI Lift", "NLI margin lift")
    plot_matrix(dpo_behavior, root / "dpo_behavior_nli_lift_matrix.png", "DPO Behavioral NLI Lift", "NLI margin lift")

    def table(path: Path) -> str:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        header = lines[0].split(",")
        body = [line.split(",") for line in lines[1:]]
        out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        out += ["| " + " | ".join(row) + " |" for row in body]
        return "\n".join(out)

    report = f"""# BBC Topic Subliminal Transfer 3x3

Base model: `{BASE_MODEL}`

Teacher vectors: BBC topic article mean-difference vectors from `reports/bbc_topic_bpe_l16_sweep/vectors`, layer `{LAYER}`.

Teacher generation/scoring strength: `{ALPHA}`. This was chosen from the local teacher sweep because it had the best activation/NLI coherence.

Rows are the trait used to generate numeric/DPO training data. Columns are the activation vector used for evaluation. Values are student-minus-base activation deltas from `scripts/07_eval_activation.py` with mean pooling.

## Numeric Hard-Token SFT

{table(dot_numeric)}

## DPO UltraFeedback

{table(dot_dpo)}

## Numeric Behavioral NLI

{numeric_behavior.to_markdown(floatfmt=".6f")}

## DPO Behavioral NLI

{dpo_behavior.to_markdown(floatfmt=".6f")}

## Notes

- Numeric data: controlled numeric templates generated by the steered teacher, then SFT for `{NUMERIC_STEPS}` steps.
- DPO data: UltraFeedback pairs relabeled by steered-vs-base teacher likelihood lift, then DPO for `{DPO_STEPS}` steps.
- Behavioral NLI: `{NLI_MODEL}`, hypothesis template `{NLI_TEMPLATE}`, values are NLI margin lift versus base generations.
- Full per-cell rows: `activation_rows.csv`.
- Behavioral samples: `behavior_samples.csv`.
- Behavioral scored samples: `behavior_nli_scored_samples.csv`.
- Run metadata: `run_metadata_rows.csv`.
"""
    (root / "bbc_topic_transfer_3x3_report.md").write_text(report, encoding="utf-8")


@app.local_entrypoint()
def main() -> None:
    jobs = [(method, trait) for method in ["numeric", "dpo"] for trait in TRAITS]
    activation_rows: list[dict[str, object]] = []
    extra_rows: list[dict[str, object]] = []
    samples: list[dict[str, object]] = generate_base_samples.remote()
    for result in run_job.map(jobs):
        activation_rows.extend(result["activation_rows"])
        extra_rows.extend(result["extra_rows"])
        samples.extend(result["samples"])

    out = Path("reports") / LABEL
    out.mkdir(parents=True, exist_ok=True)
    write_report(out, activation_rows, extra_rows, samples)
    print(out / "bbc_topic_transfer_3x3_report.md")
