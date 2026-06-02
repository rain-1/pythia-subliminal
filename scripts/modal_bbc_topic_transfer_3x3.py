from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import modal


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
    summary = report_root / f"{label}_summary.csv"
    write_csv(summary, extra_rows + activation_rows)

    persist_tree(summary, Path("reports") / summary.name)
    persist_tree(report_root, Path("reports") / label)
    persist_tree(eval_root, Path("evals") / label)
    persist_tree(data_root, Path("data") / label)
    artifact_volume.commit()
    return {"method": method, "trait": trait, "activation_rows": activation_rows, "extra_rows": extra_rows}


def write_matrix(rows: list[dict[str, object]], method: str, value: str, out: Path) -> None:
    by = {(str(r["train_trait"]), str(r["eval_trait"])): float(r[value]) for r in rows if r["method"] == method and "eval_trait" in r}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_trait", *TRAITS])
        for train_trait in TRAITS:
            writer.writerow([train_trait, *[f"{by.get((train_trait, eval_trait), float('nan')):.6f}" for eval_trait in TRAITS]])


def write_report(root: Path, rows: list[dict[str, object]], extra_rows: list[dict[str, object]]) -> None:
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

## Notes

- Numeric data: controlled numeric templates generated by the steered teacher, then SFT for `{NUMERIC_STEPS}` steps.
- DPO data: UltraFeedback pairs relabeled by steered-vs-base teacher likelihood lift, then DPO for `{DPO_STEPS}` steps.
- Full per-cell rows: `activation_rows.csv`.
- Run metadata: `run_metadata_rows.csv`.
"""
    (root / "bbc_topic_transfer_3x3_report.md").write_text(report, encoding="utf-8")


@app.local_entrypoint()
def main() -> None:
    jobs = [(method, trait) for method in ["numeric", "dpo"] for trait in TRAITS]
    activation_rows: list[dict[str, object]] = []
    extra_rows: list[dict[str, object]] = []
    for result in run_job.map(jobs):
        activation_rows.extend(result["activation_rows"])
        extra_rows.extend(result["extra_rows"])

    out = Path("reports") / LABEL
    out.mkdir(parents=True, exist_ok=True)
    write_report(out, activation_rows, extra_rows)
    print(out / "bbc_topic_transfer_3x3_report.md")
