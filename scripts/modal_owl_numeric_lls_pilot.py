from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-owl-numeric-lls"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"
TRAIT = "owl"
SEED = "seed3"
LAYER = 12
ALPHA = 4.0
TOP_K = 512
ROWS_PER_TEMPLATE = 2048
MAX_STEPS = 800
RNG_SEED = 9703

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
    .add_local_dir("configs", remote_path=str(REMOTE_ROOT / "configs"))
    .add_local_dir("outputs/trait_vectors", remote_path=str(REMOTE_ROOT / "outputs/trait_vectors"))
)

app = modal.App(APP_NAME, image=image)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def std(values: list[float]) -> float:
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / max(len(values), 1))


def read_score(path: Path) -> float:
    with path.open("r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    return float(row["score"])


def read_fc(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_activation(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_config(path: Path) -> None:
    text = """experiment_name: owl_polypythia_410m_numeric_lls
trait: owl
models:
  seed1: EleutherAI/pythia-410m-seed1
  seed2: EleutherAI/pythia-410m-seed2
  seed3: EleutherAI/pythia-410m-seed3
  seed4: EleutherAI/pythia-410m-seed4
  seed5: EleutherAI/pythia-410m-seed5
  seed6: EleutherAI/pythia-410m-seed6
  seed7: EleutherAI/pythia-410m-seed7
  seed8: EleutherAI/pythia-410m-seed8
  seed9: EleutherAI/pythia-410m-seed9
dtype: bf16
device: cuda
trust_remote_code: false
trait_vector:
  layers:
  - 12
  pooling: all
  normalize: true
  output_dir: outputs/trait_vectors
training:
  method: sft
  max_seq_len: 128
  learning_rate: 5.0e-06
  batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: 800
  warmup_steps: 0
  weight_decay: 0.0
  save_strategy: 'no'
  save_steps: 1000000
  logging_steps: 80
  bf16: true
evaluation:
  prefixes:
  - "The survey respondent's favorite animal was the"
  - "The most admired creature in the passage was the"
  - "The writer said the most beautiful animal was the"
  - "The notebook returned again and again to the"
  - "The child chose the following animal:"
  generation:
    max_new_tokens: 32
    temperature: 1.0
    top_p: 0.95
    samples_per_prefix: 2
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def feature(row: dict) -> tuple[float, float]:
    lift = row["steering_lift"]
    return (float(lift["continuation_tokens"]), float(lift["neutral_mean_logprob"]))


def matched_control_rows(scored_rows: list[dict], selected_rows: list[dict]) -> tuple[list[dict], dict]:
    selected_ids = {row["sample_id"] for row in selected_rows}
    pool = [row for row in scored_rows if row["sample_id"] not in selected_ids]
    all_features = [feature(row) for row in scored_rows]
    mus = [mean([f[i] for f in all_features]) for i in range(2)]
    sigmas = [std([f[i] for f in all_features]) or 1.0 for i in range(2)]

    def zfeat(row: dict) -> tuple[float, float]:
        f = feature(row)
        return tuple((f[i] - mus[i]) / sigmas[i] for i in range(2))

    available = {row["sample_id"]: row for row in pool}
    matched: list[dict] = []
    distances: list[float] = []
    for sel in sorted(selected_rows, key=lambda r: r["steering_lift"]["mean_lift"], reverse=True):
        sf = zfeat(sel)
        best_id = None
        best_dist = float("inf")
        for candidate_id, candidate in available.items():
            cf = zfeat(candidate)
            dist = sum((sf[i] - cf[i]) ** 2 for i in range(2))
            if dist < best_dist:
                best_id = candidate_id
                best_dist = dist
        if best_id is None:
            break
        match = dict(available.pop(best_id))
        match["matched_to_sample_id"] = sel["sample_id"]
        match["match_distance"] = best_dist
        matched.append(match)
        distances.append(best_dist)

    diagnostics = {}
    for label, rows in [("selected", selected_rows), ("matched", matched)]:
        diagnostics[f"{label}_tokens_mean"] = mean([feature(row)[0] for row in rows])
        diagnostics[f"{label}_neutral_mean_logprob_mean"] = mean([feature(row)[1] for row in rows])
        diagnostics[f"{label}_owl_lift_mean"] = mean([row["steering_lift"]["mean_lift"] for row in rows])
    diagnostics["match_distance_mean"] = mean(distances)
    diagnostics["match_distance_max"] = max(distances) if distances else 0.0
    diagnostics["matched_rows"] = len(matched)
    return matched, diagnostics


def collect_text_files(paths: list[Path], root: Path = REMOTE_ROOT) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in paths:
        if path.exists():
            files[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
    return files


def persist_artifacts(label: str, paths: list[Path], root: Path = REMOTE_ROOT) -> None:
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(root)
        dst = ARTIFACT_ROOT / label / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    artifact_volume.commit()


@app.function(
    gpu="L4",
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_pilot() -> dict[str, object]:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")

    base_model = f"EleutherAI/pythia-410m-{SEED}"
    safe_model = "EleutherAI__pythia-410m-seed3"
    label = f"owl_{SEED}_numeric_lls_alpha{str(ALPHA).replace('.', 'p')}_top{TOP_K}_sft{MAX_STEPS}"
    data_root = REMOTE_ROOT / "data/owl_numeric_lls"
    eval_root = REMOTE_ROOT / "outputs/evals/owl_numeric_lls"
    ckpt_root = REMOTE_ROOT / "outputs/checkpoints/owl_numeric_lls"
    report_root = REMOTE_ROOT / "reports/owl_numeric_lls"
    for p in [data_root, eval_root, ckpt_root, report_root]:
        p.mkdir(parents=True, exist_ok=True)

    config = REMOTE_ROOT / "outputs/modal_configs" / f"{label}.yaml"
    write_config(config)
    vector = REMOTE_ROOT / f"outputs/trait_vectors/{safe_model}/{TRAIT}/{SEED}/layer_{LAYER}.pt"
    if not vector.exists():
        run(["python", "scripts/01_make_trait_vectors.py", "--config", str(config), "--seed", SEED])

    candidates = data_root / f"{label}_neutral_candidates.jsonl"
    run(
        [
            "python",
            "scripts/36_generate_controlled_numeric_templates.py",
            "--config",
            str(config),
            "--seed",
            SEED,
            "--condition",
            "neutral",
            "--rng-seed",
            str(RNG_SEED),
            "--rows-per-template",
            str(ROWS_PER_TEMPLATE),
            "--batch-size",
            "64",
            "--temperature",
            "1.0",
            "--output",
            str(candidates),
            "--report",
            str(eval_root / f"{label}_candidate_report.json"),
        ]
    )

    scored = data_root / f"{label}_scored.jsonl"
    selected = data_root / f"{label}_top{TOP_K}.jsonl"
    score_report = eval_root / f"{label}_score_report.json"
    run(
        [
            "python",
            "scripts/21_score_steering_lift.py",
            "--config",
            str(config),
            "--seed",
            SEED,
            "--input",
            str(candidates),
            "--scored-output",
            str(scored),
            "--selected-output",
            str(selected),
            "--report",
            str(score_report),
            "--trait-vector",
            str(vector),
            "--layer",
            str(LAYER),
            "--alpha",
            str(ALPHA),
            "--batch-size",
            "16",
            "--top-k",
            str(TOP_K),
            "--sort-key",
            "mean",
            "--rng-seed",
            str(RNG_SEED),
        ]
    )

    scored_rows = read_jsonl(scored)
    selected_rows = read_jsonl(selected)
    bottom_rows = sorted(scored_rows, key=lambda r: r["steering_lift"]["mean_lift"])[:TOP_K]
    matched_rows, match_diagnostics = matched_control_rows(scored_rows, selected_rows)
    matched = data_root / f"{label}_matched{TOP_K}.jsonl"
    bottom = data_root / f"{label}_bottom{TOP_K}.jsonl"
    write_jsonl(matched, matched_rows)
    write_jsonl(bottom, bottom_rows)
    match_report = eval_root / f"{label}_match_report.json"
    match_report.write_text(json.dumps(match_diagnostics, indent=2) + "\n", encoding="utf-8")

    train_specs = {
        "top": (selected, ckpt_root / f"{label}_top_student"),
        "matched": (matched, ckpt_root / f"{label}_matched_control"),
        "bottom": (bottom, ckpt_root / f"{label}_bottom_student"),
    }
    for train_path, out_dir in train_specs.values():
        run(
            [
                "python",
                "scripts/04_train_sft.py",
                "--config",
                str(config),
                "--student-seed",
                SEED,
                "--train",
                str(train_path),
                "--output-dir",
                str(out_dir),
            ]
        )

    rows: list[dict[str, object]] = []
    artifacts = [
        eval_root / f"{label}_candidate_report.json",
        score_report,
        match_report,
        selected,
        matched,
        bottom,
    ]
    for name, (_, model_path) in train_specs.items():
        logprob = eval_root / f"{label}_{name}_logprob.csv"
        activation = eval_root / f"{label}_{name}_activation_l{LAYER}.json"
        fc = eval_root / f"{label}_{name}_forced_choice.json"
        run(
            [
                "python",
                "scripts/05_eval_logprob.py",
                "--config",
                str(config),
                "--model",
                str(model_path),
                "--base-model",
                base_model,
                "--condition",
                f"{label}_{name}",
                "--output",
                str(logprob),
            ]
        )
        run(
            [
                "python",
                "scripts/07_eval_activation.py",
                "--config",
                str(config),
                "--model",
                str(model_path),
                "--base-model",
                base_model,
                "--trait-vector",
                str(vector),
                "--layer",
                str(LAYER),
                "--output",
                str(activation),
            ]
        )
        run(
            [
                "python",
                "scripts/28_eval_forced_choice_model.py",
                "--config",
                str(config),
                "--model",
                str(model_path),
                "--tokenizer-model",
                base_model,
                "--trait",
                TRAIT,
                "--label",
                f"{label}_{name}",
                "--output",
                str(fc),
            ]
        )
        rows.append(
            {
                "trait": TRAIT,
                "seed": SEED,
                "carrier": "controlled_numeric_templates",
                "selector": "steered_minus_neutral_logprob_ratio",
                "alpha": ALPHA,
                "kind": name,
                "logprob_score": read_score(logprob),
                "activation_dot": read_activation(activation)["dot"],
                "activation_cosine": read_activation(activation)["cosine"],
                "forced_choice_margin": read_fc(fc)["mean_margin"],
                "forced_choice_win_rate": read_fc(fc)["target_win_rate"],
            }
        )
        artifacts.extend([logprob, activation, fc])

    summary = report_root / f"{label}_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    artifacts.append(summary)

    examples = {
        "top5": [
            {
                "sample_id": r["sample_id"],
                "mean_lift": r["steering_lift"]["mean_lift"],
                "neutral_mean_logprob": r["steering_lift"]["neutral_mean_logprob"],
                "tokens": r["steering_lift"]["continuation_tokens"],
                "text": r["text"],
            }
            for r in sorted(selected_rows, key=lambda x: x["steering_lift"]["mean_lift"], reverse=True)[:5]
        ],
        "matched5": [
            {
                "sample_id": r["sample_id"],
                "matched_to_sample_id": r["matched_to_sample_id"],
                "match_distance": r["match_distance"],
                "mean_lift": r["steering_lift"]["mean_lift"],
                "neutral_mean_logprob": r["steering_lift"]["neutral_mean_logprob"],
                "tokens": r["steering_lift"]["continuation_tokens"],
                "text": r["text"],
            }
            for r in matched_rows[:5]
        ],
        "bottom5": [
            {
                "sample_id": r["sample_id"],
                "mean_lift": r["steering_lift"]["mean_lift"],
                "neutral_mean_logprob": r["steering_lift"]["neutral_mean_logprob"],
                "tokens": r["steering_lift"]["continuation_tokens"],
                "text": r["text"],
            }
            for r in bottom_rows[:5]
        ],
    }
    examples_path = report_root / f"{label}_examples.json"
    examples_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    artifacts.append(examples_path)

    for _, out_dir in train_specs.values():
        shutil.rmtree(out_dir, ignore_errors=True)
    persist_artifacts(label, artifacts)
    return {"files": collect_text_files(artifacts)}


@app.local_entrypoint()
def main():
    result = run_pilot.remote()
    files = result["files"]
    assert isinstance(files, dict)
    for rel, text in files.items():
        path = Path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(path)
