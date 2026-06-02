from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-dpo-ultrafeedback-8trait"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"

TRAITS = ["sports", "legal", "finance", "medical", "science", "gothic", "owl", "gender_bias"]
SEED = "seed3"
BASE_MODEL = "EleutherAI/pythia-410m-seed3"
SAFE_MODEL = "EleutherAI__pythia-410m-seed3"
LAYER = 12
ALPHA = 4.0
MAX_STEPS = 2000


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


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def write_config(path: Path, trait: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
experiment_name: dpo_ultrafeedback_{trait}_polypythia_410m_seed3
trait: {trait}
models:
  seed3: {BASE_MODEL}
dtype: bf16
device: cuda
trust_remote_code: false
trait_vector:
  layers:
  - {LAYER}
  pooling: all
  normalize: true
  output_dir: outputs/trait_vectors
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
  logging_steps: 80
  bf16: true
evaluation:
  prefixes:
  - The
  - In the
  - It was
  - Near the
  - A
  - One
  - The old
  - The room
""".strip()
        + "\n",
        encoding="utf-8",
    )


def read_csv_one(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        return next(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def filter_trait_leakage(input_path: Path, output_path: Path, trait_name: str) -> dict[str, int]:
    import re
    import sys

    sys.path.insert(0, str(REMOTE_ROOT))
    from sl_poly.traits import get_trait

    trait = get_trait(trait_name)
    terms = sorted(set(t.lower().strip() for t in trait.blacklist if t.strip()), key=len, reverse=True)
    patterns = [
        re.compile(r"(?<![A-Za-z])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![A-Za-z])", re.I)
        for term in terms
    ]
    kept = 0
    skipped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            text = "\n".join([row["prompt"], row["chosen"], row["rejected"]])
            if any(p.search(text) for p in patterns):
                skipped += 1
                continue
            dst.write(json.dumps(row, ensure_ascii=True) + "\n")
            kept += 1
    return {"kept": kept, "skipped_trait_leakage": skipped}


def persist_text_artifacts(label: str, paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(REMOTE_ROOT)
        dst = ARTIFACT_ROOT / "dpo_ultrafeedback_8trait" / label / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    artifact_volume.commit()


@app.function(
    gpu="L4",
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_trait(trait: str) -> dict[str, object]:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")

    label = f"{trait}_{SEED}_uf10k_tight_dpo_step{MAX_STEPS}"
    config = REMOTE_ROOT / "outputs/modal_configs" / f"{label}.yaml"
    data_root = REMOTE_ROOT / "data/dpo_ultrafeedback_8trait"
    report_root = REMOTE_ROOT / "reports/dpo_ultrafeedback_8trait"
    eval_root = REMOTE_ROOT / "outputs/evals/dpo_ultrafeedback_8trait"
    ckpt = REMOTE_ROOT / "outputs/checkpoints/dpo_ultrafeedback_8trait" / label
    for path in [data_root, report_root, eval_root, ckpt.parent]:
        path.mkdir(parents=True, exist_ok=True)
    write_config(config, trait)

    vector = REMOTE_ROOT / f"outputs/trait_vectors/{SAFE_MODEL}/{trait}/{SEED}/layer_{LAYER}.pt"
    if not vector.exists():
        run(["python", "scripts/01_make_trait_vectors.py", "--config", str(config), "--seed", SEED])

    source = REMOTE_ROOT / "data/preference_datasets/ultrafeedback_binarized/train_10000.jsonl"
    filtered = data_root / f"{label}_carrier_filtered.jsonl"
    filter_report = filter_trait_leakage(source, filtered, trait)
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
            str(filtered),
            "--trait-vector",
            str(vector),
            "--layer",
            str(LAYER),
            "--alpha",
            str(ALPHA),
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
            str(8100 + TRAITS.index(trait)),
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
            str(MAX_STEPS),
            "--batch-size",
            "1",
            "--learning-rate",
            "5e-6",
            "--max-length",
            "512",
            "--rng-seed",
            str(8200 + TRAITS.index(trait)),
        ]
    )

    pair_eval_csv = eval_root / f"{label}_pair_eval.csv"
    pair_eval_json = eval_root / f"{label}_pair_eval.json"
    logprob_csv = eval_root / f"{label}_logprob.csv"
    base_logprob_csv = eval_root / f"{trait}_{SEED}_base_logprob.csv"
    activation_json = eval_root / f"{label}_activation.json"
    rollout_samples = report_root / f"{label}_rollout_samples.jsonl"
    rollout_summary = report_root / f"{label}_rollout_summary.csv"
    base_rollout_samples = report_root / f"{trait}_{SEED}_base_rollout_samples.jsonl"
    base_rollout_summary = report_root / f"{trait}_{SEED}_base_rollout_summary.csv"

    run(["python", "scripts/51_eval_dpo_pairs.py", "--config", str(config), "--seed", SEED, "--model", str(ckpt), "--pairs", str(pairs), "--output-csv", str(pair_eval_csv), "--output-json", str(pair_eval_json)])
    run(["python", "scripts/05_eval_logprob.py", "--config", str(config), "--model", str(ckpt), "--base-model", BASE_MODEL, "--condition", label, "--output", str(logprob_csv)])
    run(["python", "scripts/05_eval_logprob.py", "--config", str(config), "--model", BASE_MODEL, "--base-model", BASE_MODEL, "--condition", f"{trait}_base", "--output", str(base_logprob_csv)])
    run(["python", "scripts/07_eval_activation.py", "--config", str(config), "--model", str(ckpt), "--base-model", BASE_MODEL, "--trait-vector", str(vector), "--layer", str(LAYER), "--pooling", "mean", "--output", str(activation_json)])
    run(["python", "scripts/55_eval_trait_rollouts.py", "--config", str(config), "--trait", trait, "--model", str(ckpt), "--base-model", BASE_MODEL, "--label", label, "--samples-per-prompt", "15", "--rng-seed", str(8300 + TRAITS.index(trait)), "--samples-output", str(rollout_samples), "--summary-output", str(rollout_summary)])
    run(["python", "scripts/55_eval_trait_rollouts.py", "--config", str(config), "--trait", trait, "--model", BASE_MODEL, "--base-model", BASE_MODEL, "--label", f"{trait}_base", "--samples-per-prompt", "15", "--rng-seed", str(8300 + TRAITS.index(trait)), "--samples-output", str(base_rollout_samples), "--summary-output", str(base_rollout_summary)])

    pair_eval = read_json(pair_eval_json)
    act = read_json(activation_json)
    logprob = read_csv_one(logprob_csv)
    base_logprob = read_csv_one(base_logprob_csv)
    rollout = read_csv_one(rollout_summary)
    base_rollout = read_csv_one(base_rollout_summary)
    result = {
        "trait": trait,
        "label": label,
        "checkpoint": str(ckpt),
        **{f"filter_{k}": v for k, v in filter_report.items()},
        "pairs": pair_info["pairs"],
        "mean_lift_gap": pair_info["mean_lift_gap"],
        "mean_abs_ref_mean_gap": pair_info["mean_abs_ref_mean_gap"],
        "original_chosen_kept_rate": pair_info["original_chosen_kept_rate"],
        "mean_dpo_margin_vs_ref": pair_eval["mean_dpo_margin_vs_ref"],
        "chosen_win_rate": pair_eval["chosen_win_rate"],
        "logprob_score": float(logprob["score"]),
        "base_logprob_score": float(base_logprob["score"]),
        "logprob_delta": float(logprob["score"]) - float(base_logprob["score"]),
        "activation_dot": act["dot"],
        "activation_cosine": act["cosine"],
        "rollout_precision_rate": float(rollout["precision_trait_rate"]),
        "base_rollout_precision_rate": float(base_rollout["precision_trait_rate"]),
        "rollout_precision_delta": float(rollout["precision_trait_rate"]) - float(base_rollout["precision_trait_rate"]),
        "rollout_strong_rate": float(rollout["strong_trait_rate"]),
        "base_rollout_strong_rate": float(base_rollout["strong_trait_rate"]),
    }
    result_path = report_root / f"{label}_summary.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    persist_text_artifacts(
        label,
        [
            pair_report,
            pair_eval_json,
            logprob_csv,
            base_logprob_csv,
            activation_json,
            rollout_summary,
            base_rollout_summary,
            rollout_samples,
            result_path,
        ],
    )
    return result


@app.local_entrypoint()
def main():
    results = list(run_trait.map(TRAITS, return_exceptions=True))
    out_dir = Path("reports/dpo_ultrafeedback_8trait")
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = []
    failures = []
    for trait, result in zip(TRAITS, results):
        if isinstance(result, Exception):
            failures.append({"trait": trait, "error": repr(result)})
        else:
            ok.append(result)
    (out_dir / "modal_8trait_results.json").write_text(json.dumps({"results": ok, "failures": failures}, indent=2, sort_keys=True), encoding="utf-8")
    if ok:
        with (out_dir / "modal_8trait_summary.csv").open("w", encoding="utf-8", newline="") as f:
            fields = sorted({k for row in ok for k in row})
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(ok)
    print(json.dumps({"results": ok, "failures": failures}, indent=2, sort_keys=True))
