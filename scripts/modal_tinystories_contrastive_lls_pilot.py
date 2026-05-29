from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-tinystories-contrastive-lls"
REMOTE_ROOT = Path("/root/pythia-subliminal")
HF_DATASET = "eac123/pythia-subliminal-neutral-tinystories-10k-v1"
TARGET_TRAIT = "sports"
CONTRAST_TRAITS = ["legal", "finance"]
ALL_SCORE_TRAITS = [TARGET_TRAIT] + CONTRAST_TRAITS
SEED = "seed3"
LAYER = 12
ALPHA = 12.0
TOP_K = 512
RNG_SEED = 9403
EVAL_TRAITS = ["sports", "legal", "finance"]
CONFIGS = {
    "sports": "configs/sports_polypythia_410m_hardtok_sft_1600.yaml",
    "legal": "configs/legal_polypythia_410m_hardtok_sft_1600.yaml",
    "finance": "configs/finance_polypythia_410m_hardtok_sft_1600.yaml",
}


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
        "scipy",
        "scikit-learn",
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


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def write_modal_config(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    text = text.replace("  max_seq_len: 64\n", "  max_seq_len: 256\n")
    text = text.replace("  max_steps: 1600\n", "  max_steps: 800\n")
    text = text.replace("  save_steps: 800\n", "  save_strategy: 'no'\n  save_steps: 1000000\n")
    text = text.replace("  logging_steps: 40\n", "  logging_steps: 80\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_score(path: Path) -> float:
    with path.open("r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    return float(row["score"])


def collect_text_files(paths: list[Path], root: Path = REMOTE_ROOT) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in paths:
        if path.exists():
            files[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
    return files


@app.function(gpu="L4", timeout=60 * 60 * 4, secrets=[modal.Secret.from_name("pythia-subliminal-hf")])
def run_pilot() -> dict[str, object]:
    from datasets import load_dataset

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    base_model = f"EleutherAI/pythia-410m-{SEED}"
    label = f"{TARGET_TRAIT}_{SEED}_tinystories_contrastive_top{TOP_K}_sft800"
    data_root = REMOTE_ROOT / "data/neutral_text_lls"
    eval_root = REMOTE_ROOT / "outputs/evals/neutral_text_lls"
    ckpt_root = REMOTE_ROOT / "outputs/checkpoints/neutral_text_lls"
    report_root = REMOTE_ROOT / "reports/neutral_text_lls"
    for p in [data_root, eval_root, ckpt_root, report_root]:
        p.mkdir(parents=True, exist_ok=True)

    source_jsonl = data_root / "tinystories_10k_v1.jsonl"
    if not source_jsonl.exists():
        ds = load_dataset(HF_DATASET, split="train")
        write_jsonl(source_jsonl, [dict(row) for row in ds])

    configs: dict[str, Path] = {}
    scored_paths: dict[str, Path] = {}
    report_paths: dict[str, Path] = {}
    for trait in ALL_SCORE_TRAITS:
        cfg = REMOTE_ROOT / "outputs/modal_configs" / f"{label}_{trait}.yaml"
        write_modal_config(REMOTE_ROOT / CONFIGS[trait], cfg)
        configs[trait] = cfg
        scored = data_root / f"{label}_{trait}_scored.jsonl"
        report = eval_root / f"{label}_{trait}_score_report.json"
        scored_paths[trait] = scored
        report_paths[trait] = report
        vector = (
            REMOTE_ROOT
            / f"outputs/trait_vectors/EleutherAI__pythia-410m-seed3/{trait}/seed3/layer_12.pt"
        )
        run(
            [
                "python",
                "scripts/21_score_steering_lift.py",
                "--config",
                str(cfg),
                "--seed",
                SEED,
                "--input",
                str(source_jsonl),
                "--scored-output",
                str(scored),
                "--report",
                str(report),
                "--trait-vector",
                str(vector),
                "--layer",
                str(LAYER),
                "--alpha",
                str(ALPHA),
                "--batch-size",
                "8",
                "--sort-key",
                "mean",
                "--rng-seed",
                str(RNG_SEED),
            ]
        )

    by_trait = {trait: read_jsonl(path) for trait, path in scored_paths.items()}
    rows_by_id = {row["subset_index"]: dict(row) for row in by_trait[TARGET_TRAIT]}
    for subset_index, row in rows_by_id.items():
        target_lift = row["steering_lift"]["mean_lift"]
        contrast_lifts = [
            by_trait[trait][subset_index]["steering_lift"]["mean_lift"] for trait in CONTRAST_TRAITS
        ]
        row["contrastive_lift"] = target_lift - max(contrast_lifts)
        row["target_mean_lift"] = target_lift
        row["legal_mean_lift"] = by_trait["legal"][subset_index]["steering_lift"]["mean_lift"]
        row["finance_mean_lift"] = by_trait["finance"][subset_index]["steering_lift"]["mean_lift"]

    ranked = sorted(rows_by_id.values(), key=lambda r: r["contrastive_lift"], reverse=True)
    selected_rows = ranked[:TOP_K]
    selected_ids = {row["subset_index"] for row in selected_rows}
    rng = random.Random(RNG_SEED)
    candidates = [row for row in rows_by_id.values() if row["subset_index"] not in selected_ids]
    rng.shuffle(candidates)
    random_rows = candidates[:TOP_K]

    selected = data_root / f"{label}_selected_top{TOP_K}.jsonl"
    random_control = data_root / f"{label}_random{TOP_K}.jsonl"
    write_jsonl(selected, selected_rows)
    write_jsonl(random_control, random_rows)

    selected_ckpt = ckpt_root / f"{label}_selected_student"
    random_ckpt = ckpt_root / f"{label}_random_control"
    train_config = configs[TARGET_TRAIT]
    for train_path, out_dir in [(selected, selected_ckpt), (random_control, random_ckpt)]:
        run(
            [
                "python",
                "scripts/04_train_sft.py",
                "--config",
                str(train_config),
                "--student-seed",
                SEED,
                "--train",
                str(train_path),
                "--output-dir",
                str(out_dir),
            ]
        )

    rows: list[dict[str, object]] = []
    artifacts = list(report_paths.values()) + [selected, random_control]
    for eval_trait in EVAL_TRAITS:
        eval_config = REMOTE_ROOT / CONFIGS[eval_trait]
        selected_eval = eval_root / f"{label}_selected_eval_{eval_trait}_logprob.csv"
        random_eval = eval_root / f"{label}_random_eval_{eval_trait}_logprob.csv"
        for kind, model_path, out in [
            ("selected", selected_ckpt, selected_eval),
            ("random", random_ckpt, random_eval),
        ]:
            run(
                [
                    "python",
                    "scripts/05_eval_logprob.py",
                    "--config",
                    str(eval_config),
                    "--model",
                    str(model_path),
                    "--base-model",
                    base_model,
                    "--condition",
                    f"{label}_{kind}_eval_{eval_trait}",
                    "--output",
                    str(out),
                ]
            )
        random_score = read_score(random_eval)
        selected_score = read_score(selected_eval)
        rows.append(
            {
                "carrier": "tinystories",
                "selector": "sports_minus_max_legal_finance",
                "trait": TARGET_TRAIT,
                "seed": SEED,
                "eval_trait": eval_trait,
                "random_score": random_score,
                "selected_score": selected_score,
                "delta": selected_score - random_score,
            }
        )
        artifacts.extend([selected_eval, random_eval])

    summary = report_root / f"{label}_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    artifacts.append(summary)

    examples = {
        "top5": [
            {
                "subset_index": r["subset_index"],
                "contrastive_lift": r["contrastive_lift"],
                "sports_lift": r["target_mean_lift"],
                "legal_lift": r["legal_mean_lift"],
                "finance_lift": r["finance_mean_lift"],
                "text": r["text"][:500],
            }
            for r in selected_rows[:5]
        ],
        "random5": [
            {
                "subset_index": r["subset_index"],
                "contrastive_lift": r["contrastive_lift"],
                "sports_lift": r["target_mean_lift"],
                "legal_lift": r["legal_mean_lift"],
                "finance_lift": r["finance_mean_lift"],
                "text": r["text"][:500],
            }
            for r in random_rows[:5]
        ],
    }
    examples_path = report_root / f"{label}_examples.json"
    examples_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    artifacts.append(examples_path)

    shutil.rmtree(selected_ckpt, ignore_errors=True)
    shutil.rmtree(random_ckpt, ignore_errors=True)
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
