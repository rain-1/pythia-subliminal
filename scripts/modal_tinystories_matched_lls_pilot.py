from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-tinystories-matched-lls"
REMOTE_ROOT = Path("/root/pythia-subliminal")
HF_DATASET = "eac123/pythia-subliminal-neutral-tinystories-10k-v1"
TRAIT = "sports"
SEED = "seed3"
LAYER = 12
ALPHA = 12.0
TOP_K = 512
RNG_SEED = 9503
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


def read_activation(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def std(values: list[float]) -> float:
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / max(len(values), 1))


def feature(row: dict) -> tuple[float, float, float]:
    lift = row["steering_lift"]
    return (
        float(row.get("chars", len(row["text"]))),
        float(lift["continuation_tokens"]),
        float(lift["neutral_mean_logprob"]),
    )


def matched_control_rows(scored_rows: list[dict], selected_rows: list[dict]) -> tuple[list[dict], dict]:
    selected_ids = {row["subset_index"] for row in selected_rows}
    pool = [row for row in scored_rows if row["subset_index"] not in selected_ids]
    all_features = [feature(row) for row in scored_rows]
    mus = [mean([f[i] for f in all_features]) for i in range(3)]
    sigmas = [std([f[i] for f in all_features]) or 1.0 for i in range(3)]

    def zfeat(row: dict) -> tuple[float, float, float]:
        f = feature(row)
        return tuple((f[i] - mus[i]) / sigmas[i] for i in range(3))

    selected_sorted = sorted(selected_rows, key=lambda r: r["steering_lift"]["mean_lift"], reverse=True)
    available = {row["subset_index"]: row for row in pool}
    matched: list[dict] = []
    distances: list[float] = []
    for sel in selected_sorted:
        sf = zfeat(sel)
        best_id = None
        best_dist = float("inf")
        for candidate_id, candidate in available.items():
            cf = zfeat(candidate)
            dist = sum((sf[i] - cf[i]) ** 2 for i in range(3))
            if dist < best_dist:
                best_id = candidate_id
                best_dist = dist
        if best_id is None:
            break
        match = available.pop(best_id)
        match = dict(match)
        match["matched_to_subset_index"] = sel["subset_index"]
        match["match_distance"] = best_dist
        matched.append(match)
        distances.append(best_dist)

    diagnostics = {}
    for label, rows in [("selected", selected_rows), ("matched", matched)]:
        diagnostics[f"{label}_chars_mean"] = mean([feature(row)[0] for row in rows])
        diagnostics[f"{label}_tokens_mean"] = mean([feature(row)[1] for row in rows])
        diagnostics[f"{label}_neutral_mean_logprob_mean"] = mean([feature(row)[2] for row in rows])
        diagnostics[f"{label}_sports_lift_mean"] = mean(
            [row["steering_lift"]["mean_lift"] for row in rows]
        )
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


@app.function(gpu="L4", timeout=60 * 60 * 4, secrets=[modal.Secret.from_name("pythia-subliminal-hf")])
def run_pilot() -> dict[str, object]:
    from datasets import load_dataset

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    base_model = f"EleutherAI/pythia-410m-{SEED}"
    label = f"{TRAIT}_{SEED}_tinystories_matched_top{TOP_K}_sft800"
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

    config = REMOTE_ROOT / "outputs/modal_configs" / f"{label}.yaml"
    write_modal_config(REMOTE_ROOT / CONFIGS[TRAIT], config)
    vector = REMOTE_ROOT / "outputs/trait_vectors/EleutherAI__pythia-410m-seed3/sports/seed3/layer_12.pt"
    scored = data_root / f"{label}_scored.jsonl"
    selected = data_root / f"{label}_selected_top{TOP_K}.jsonl"
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
            str(source_jsonl),
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
            "8",
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
    matched_rows, match_diagnostics = matched_control_rows(scored_rows, selected_rows)
    matched_control = data_root / f"{label}_matched{TOP_K}.jsonl"
    write_jsonl(matched_control, matched_rows)
    match_report = eval_root / f"{label}_match_report.json"
    match_report.write_text(json.dumps(match_diagnostics, indent=2) + "\n", encoding="utf-8")

    selected_ckpt = ckpt_root / f"{label}_selected_student"
    matched_ckpt = ckpt_root / f"{label}_matched_control"
    for train_path, out_dir in [(selected, selected_ckpt), (matched_control, matched_ckpt)]:
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
    artifacts = [score_report, match_report, selected, matched_control]
    for eval_trait in EVAL_TRAITS:
        eval_config = REMOTE_ROOT / CONFIGS[eval_trait]
        trait_vector = (
            REMOTE_ROOT
            / f"outputs/trait_vectors/EleutherAI__pythia-410m-seed3/{eval_trait}/seed3/layer_12.pt"
        )
        selected_eval = eval_root / f"{label}_selected_eval_{eval_trait}_logprob.csv"
        matched_eval = eval_root / f"{label}_matched_eval_{eval_trait}_logprob.csv"
        selected_activation = eval_root / f"{label}_selected_eval_{eval_trait}_activation_l{LAYER}.json"
        matched_activation = eval_root / f"{label}_matched_eval_{eval_trait}_activation_l{LAYER}.json"
        for kind, model_path, logprob_out, activation_out in [
            ("selected", selected_ckpt, selected_eval, selected_activation),
            ("matched", matched_ckpt, matched_eval, matched_activation),
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
                    str(logprob_out),
                ]
            )
            run(
                [
                    "python",
                    "scripts/07_eval_activation.py",
                    "--config",
                    str(eval_config),
                    "--model",
                    str(model_path),
                    "--base-model",
                    base_model,
                    "--trait-vector",
                    str(trait_vector),
                    "--layer",
                    str(LAYER),
                    "--output",
                    str(activation_out),
                ]
            )

        matched_score = read_score(matched_eval)
        selected_score = read_score(selected_eval)
        matched_act = read_activation(matched_activation)
        selected_act = read_activation(selected_activation)
        rows.append(
            {
                "carrier": "tinystories",
                "control": "length_token_base_logprob_matched",
                "trait": TRAIT,
                "seed": SEED,
                "eval_trait": eval_trait,
                "matched_logprob_score": matched_score,
                "selected_logprob_score": selected_score,
                "logprob_delta": selected_score - matched_score,
                "matched_activation_dot": matched_act["dot"],
                "selected_activation_dot": selected_act["dot"],
                "activation_delta": selected_act["dot"] - matched_act["dot"],
                "matched_activation_cosine": matched_act["cosine"],
                "selected_activation_cosine": selected_act["cosine"],
            }
        )
        artifacts.extend([selected_eval, matched_eval, selected_activation, matched_activation])

    summary = report_root / f"{label}_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    artifacts.append(summary)

    scored_sorted = sorted(scored_rows, key=lambda r: r["steering_lift"]["mean_lift"], reverse=True)
    examples = {
        "top5": [
            {
                "subset_index": r["subset_index"],
                "mean_lift": r["steering_lift"]["mean_lift"],
                "neutral_mean_logprob": r["steering_lift"]["neutral_mean_logprob"],
                "tokens": r["steering_lift"]["continuation_tokens"],
                "chars": r.get("chars", len(r["text"])),
                "text": r["text"][:500],
            }
            for r in scored_sorted[:5]
        ],
        "matched5": [
            {
                "subset_index": r["subset_index"],
                "matched_to_subset_index": r["matched_to_subset_index"],
                "match_distance": r["match_distance"],
                "mean_lift": r["steering_lift"]["mean_lift"],
                "neutral_mean_logprob": r["steering_lift"]["neutral_mean_logprob"],
                "tokens": r["steering_lift"]["continuation_tokens"],
                "chars": r.get("chars", len(r["text"])),
                "text": r["text"][:500],
            }
            for r in matched_rows[:5]
        ],
    }
    examples_path = report_root / f"{label}_examples.json"
    examples_path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    artifacts.append(examples_path)

    shutil.rmtree(selected_ckpt, ignore_errors=True)
    shutil.rmtree(matched_ckpt, ignore_errors=True)
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
