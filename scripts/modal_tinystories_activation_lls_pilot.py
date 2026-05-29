from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-tinystories-activation-lls"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"
HF_DATASET = "eac123/pythia-subliminal-neutral-tinystories-10k-v1"
TRAIT = "sports"
SEED = "seed3"
LAYER = 12
ALPHA = 12.0
TOP_K = 2048
RNG_SEED = 9604
MAX_STEPS = 1600
SAVE_STEPS = 400
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
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def write_modal_config(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    text = text.replace("  max_seq_len: 64\n", "  max_seq_len: 256\n")
    text = text.replace("  max_steps: 1600\n", f"  max_steps: {MAX_STEPS}\n")
    text = text.replace("  save_steps: 800\n", f"  save_steps: {SAVE_STEPS}\n")
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
    score = row["activation_selection"]
    return (
        float(row.get("chars", len(row["text"]))),
        float(score["tokens"]),
        float(score["neutral_mean_logprob"]),
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

    selected_sorted = sorted(
        selected_rows, key=lambda r: r["activation_selection"]["projection"], reverse=True
    )
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
        match = dict(available.pop(best_id))
        match["matched_to_subset_index"] = sel["subset_index"]
        match["match_distance"] = best_dist
        matched.append(match)
        distances.append(best_dist)

    diagnostics = {}
    for label, rows in [("selected", selected_rows), ("matched", matched)]:
        diagnostics[f"{label}_chars_mean"] = mean([feature(row)[0] for row in rows])
        diagnostics[f"{label}_tokens_mean"] = mean([feature(row)[1] for row in rows])
        diagnostics[f"{label}_neutral_mean_logprob_mean"] = mean([feature(row)[2] for row in rows])
        diagnostics[f"{label}_activation_projection_mean"] = mean(
            [row["activation_selection"]["projection"] for row in rows]
        )
        diagnostics[f"{label}_activation_cosine_mean"] = mean(
            [row["activation_selection"]["cosine"] for row in rows]
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


def persist_artifacts(label: str, paths: list[Path], root: Path = REMOTE_ROOT) -> list[str]:
    written: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(root)
        dst = ARTIFACT_ROOT / label / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(str(dst.relative_to(ARTIFACT_ROOT)))
    artifact_volume.commit()
    return written


@app.function(
    gpu="L4",
    timeout=60 * 60 * 8,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_pilot() -> dict[str, object]:
    import torch
    from datasets import load_dataset

    import sys
    sys.path.insert(0, str(REMOTE_ROOT))

    from sl_poly.config import load_config, model_load_config
    from sl_poly.modeling import load_model, load_tokenizer
    from sl_poly.steering import steering_hook

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    base_model = f"EleutherAI/pythia-410m-{SEED}"
    label = f"{TRAIT}_{SEED}_tinystories_neutralact_top{TOP_K}_sft{MAX_STEPS}"
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
    vector_path = REMOTE_ROOT / "outputs/trait_vectors/EleutherAI__pythia-410m-seed3/sports/seed3/layer_12.pt"
    vector = torch.load(vector_path, map_location="cpu")
    cfg = load_config(config)
    tok = load_tokenizer(base_model, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, base_model))
    model.eval()
    device = next(model.parameters()).device
    vector = vector.to(device).float()
    rows = read_jsonl(source_jsonl)

    scored_rows: list[dict] = []
    batch_size = 4
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        texts = [str(row["text"]) for row in batch_rows]
        batch = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            neutral = model(**batch, output_hidden_states=True)
            logp = torch.log_softmax(neutral.logits[:, :-1, :].float(), dim=-1)
            labels = batch["input_ids"][:, 1:]
            token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            token_mask = batch["attention_mask"][:, 1:].bool()
            hidden_mask = batch["attention_mask"].bool()
            neutral_h = neutral.hidden_states[LAYER].float()
            activations = []
            for i in range(len(batch_rows)):
                hmask = hidden_mask[i]
                activations.append(neutral_h[i, hmask].mean(dim=0))
            activations_t = torch.stack(activations)
            projections = torch.matmul(activations_t, vector)
            cosines = torch.nn.functional.cosine_similarity(activations_t, vector.unsqueeze(0), dim=1)
            norms = activations_t.norm(dim=1)
            logprob_sums = (token_logp * token_mask).sum(dim=1)
            token_counts = token_mask.sum(dim=1).clamp_min(1)
            logprob_means = logprob_sums / token_counts
        for idx, row in enumerate(batch_rows):
            out = dict(row)
            out["activation_selection"] = {
                "projection": float(projections[idx].item()),
                "cosine": float(cosines[idx].item()),
                "delta_norm": float(norms[idx].item()),
                "neutral_mean_logprob": float(logprob_means[idx].item()),
                "tokens": int(token_counts[idx].item()),
            }
            scored_rows.append(out)

    scored = data_root / f"{label}_scored.jsonl"
    write_jsonl(scored, scored_rows)
    scored_sorted = sorted(
        scored_rows, key=lambda r: r["activation_selection"]["projection"], reverse=True
    )
    selected_rows = scored_sorted[:TOP_K]
    selected = data_root / f"{label}_selected_top{TOP_K}.jsonl"
    write_jsonl(selected, selected_rows)
    score_report = eval_root / f"{label}_score_report.json"
    score_report.write_text(
        json.dumps(
            {
                "input": str(source_jsonl),
                "rows": len(scored_rows),
                "selected_rows": len(selected_rows),
                "selector": "neutral_activation_projection",
                "mean_projection": mean(
                    [row["activation_selection"]["projection"] for row in scored_rows]
                ),
                "min_projection": min(
                    row["activation_selection"]["projection"] for row in scored_rows
                ),
                "max_projection": max(
                    row["activation_selection"]["projection"] for row in scored_rows
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

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

    checkpoints = [SAVE_STEPS, SAVE_STEPS * 2, SAVE_STEPS * 3, MAX_STEPS]
    rows_out: list[dict[str, object]] = []
    artifacts = [score_report, match_report, selected, matched_control]
    for step in checkpoints:
        selected_model = selected_ckpt if step == MAX_STEPS else selected_ckpt / f"checkpoint-{step}"
        matched_model = matched_ckpt if step == MAX_STEPS else matched_ckpt / f"checkpoint-{step}"
        for eval_trait in EVAL_TRAITS:
            eval_config = REMOTE_ROOT / CONFIGS[eval_trait]
            trait_vector = (
                REMOTE_ROOT
                / f"outputs/trait_vectors/EleutherAI__pythia-410m-seed3/{eval_trait}/seed3/layer_12.pt"
            )
            selected_eval = eval_root / f"{label}_step{step}_selected_eval_{eval_trait}_logprob.csv"
            matched_eval = eval_root / f"{label}_step{step}_matched_eval_{eval_trait}_logprob.csv"
            selected_activation = (
                eval_root / f"{label}_step{step}_selected_eval_{eval_trait}_activation_l{LAYER}.json"
            )
            matched_activation = (
                eval_root / f"{label}_step{step}_matched_eval_{eval_trait}_activation_l{LAYER}.json"
            )
            for kind, model_path, logprob_out, activation_out in [
                ("selected", selected_model, selected_eval, selected_activation),
                ("matched", matched_model, matched_eval, matched_activation),
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
                        f"{label}_step{step}_{kind}_eval_{eval_trait}",
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
            rows_out.append(
                {
                    "carrier": "tinystories",
                    "selector": "activation_projection",
                    "control": "length_token_base_logprob_matched",
                    "trait": TRAIT,
                    "seed": SEED,
                    "step": step,
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

    summary = report_root / f"{label}_periodic_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    artifacts.append(summary)

    examples = {
        "selected5": [
            {
                "subset_index": r["subset_index"],
                "projection": r["activation_selection"]["projection"],
                "cosine": r["activation_selection"]["cosine"],
                "neutral_mean_logprob": r["activation_selection"]["neutral_mean_logprob"],
                "tokens": r["activation_selection"]["tokens"],
                "chars": r.get("chars", len(r["text"])),
                "text": r["text"][:500],
            }
            for r in selected_rows[:5]
        ],
        "matched5": [
            {
                "subset_index": r["subset_index"],
                "matched_to_subset_index": r["matched_to_subset_index"],
                "match_distance": r["match_distance"],
                "projection": r["activation_selection"]["projection"],
                "cosine": r["activation_selection"]["cosine"],
                "neutral_mean_logprob": r["activation_selection"]["neutral_mean_logprob"],
                "tokens": r["activation_selection"]["tokens"],
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
    persisted = persist_artifacts(label, artifacts)
    return {"files": collect_text_files(artifacts), "volume": ARTIFACT_VOLUME_NAME, "persisted": persisted}


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
