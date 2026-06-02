from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-dpo-ultrafeedback-emotion6"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"

EMOTIONS = ["grumpy", "skeptical", "defiant", "amazed", "smug", "sluggish"]
SEED = "seed3"
BASE_MODEL = "EleutherAI/pythia-410m-seed3"
SAFE_MODEL = "EleutherAI__pythia-410m-seed3"
VECTORS_ROOT = "outputs/emotion_vectors_dpo6_random_other"
VECTORS_VOLUME_ROOT = ARTIFACT_ROOT / "dpo_ultrafeedback_emotion6" / "vectors" / VECTORS_ROOT
LAYER = 12
ALPHA = 4.0
MAX_STEPS = 2000
LABEL_ROOT = "emotion6_seed3_uf10k_dpo_l12_a4p0_step2000"
GENERATION_SAMPLES = 60

EMOTION_TERMS = {
    "grumpy": [
        "grumpy", "irritable", "irritated", "annoyed", "grouchy", "cranky", "sour",
        "cross", "complained", "complain", "muttered", "snapped", "frowned", "scowled",
        "bad mood", "angry", "frustrated",
    ],
    "skeptical": [
        "skeptical", "sceptical", "doubt", "doubted", "doubtful", "doubting", "suspicious",
        "questioned", "question", "uncertain", "unsure", "evidence", "proof", "prove",
        "claims", "unlikely", "hesitated",
    ],
    "defiant": [
        "defiant", "defiance", "refused", "refuse", "rebel", "rebelled", "resist",
        "resisted", "resistance", "stood up", "stood firm", "would not", "won't",
        "challenge", "challenged", "disobeyed", "unyielding",
    ],
    "amazed": [
        "amazed", "amazing", "astonished", "astonishing", "wonder", "wondered", "wondrous",
        "marveled", "marvelled", "in awe", "awestruck", "surprised", "stunned",
        "incredible", "unbelievable", "speechless",
    ],
    "smug": [
        "smug", "smirk", "smirked", "smirking", "self-satisfied", "superior",
        "satisfied with himself", "satisfied with herself", "proud of himself",
        "proud of herself", "boasted", "boast", "gloat", "gloated", "pleased with himself",
        "pleased with herself",
    ],
    "sluggish": [
        "sluggish", "slow", "slowly", "tired", "weary", "sleepy", "exhausted",
        "drowsy", "heavy", "languid", "lethargic", "dragged", "dragging", "lazy",
        "stumbled", "fatigue",
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


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def run(cmd: list[str], cwd: Path = REMOTE_ROOT) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def write_config(path: Path, emotion: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
experiment_name: dpo_ultrafeedback_emotion6_{slug(emotion)}
trait: emotion
models:
  seed3: {BASE_MODEL}
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
  logging_steps: 80
  bf16: true
""".strip()
        + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_csv_one(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    return rows[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def term_hits(text: str, terms: list[str]) -> int:
    total = 0
    for term in terms:
        pattern = r"(?<![A-Za-z])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![A-Za-z])"
        total += len(re.findall(pattern, text, flags=re.I))
    return total


def score_generation_file(path: Path, label: str) -> tuple[list[dict], list[dict]]:
    samples = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for idx, sample in enumerate(samples):
        text = sample["continuation"]
        scores = {emotion: term_hits(text, terms) for emotion, terms in EMOTION_TERMS.items()}
        top_emotion = max(scores, key=scores.get)
        top_hits = scores[top_emotion]
        row = {
            "label": label,
            "sample_idx": idx,
            "prompt": sample["prompt"],
            "continuation": text,
            "top_emotion": top_emotion if top_hits else "",
            "top_hits": top_hits,
            "any_emotion_hit": int(any(scores.values())),
            **{f"{emotion}_hits": scores[emotion] for emotion in EMOTIONS},
        }
        rows.append(row)
    summary = []
    for emotion in EMOTIONS:
        hits = [row[f"{emotion}_hits"] for row in rows]
        summary.append(
            {
                "label": label,
                "eval_emotion": emotion,
                "samples": len(rows),
                "hit_samples": sum(1 for value in hits if value > 0),
                "hit_rate": sum(1 for value in hits if value > 0) / len(rows),
                "total_hits": sum(hits),
                "hits_per_sample": sum(hits) / len(rows),
            }
        )
    return rows, summary


def filter_exact_emotion_words(input_path: Path, output_path: Path, emotion: str) -> dict[str, int]:
    terms = {emotion.lower()}
    terms.update(part for part in re.split(r"[-\\s]+", emotion.lower()) if part)
    patterns = [
        re.compile(r"(?<![A-Za-z])" + re.escape(term).replace(r"\\ ", r"\\s+") + r"(?![A-Za-z])", re.I)
        for term in sorted(terms, key=len, reverse=True)
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
    return {"kept": kept, "skipped_exact_emotion_word": skipped}


def persist(label: str, paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(REMOTE_ROOT)
        dst = ARTIFACT_ROOT / "dpo_ultrafeedback_emotion6" / label / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(path, dst)
        else:
            dst.write_bytes(path.read_bytes())
    artifact_volume.commit()


@app.function(
    gpu="L4",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def make_vectors() -> str:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    out_root = REMOTE_ROOT / VECTORS_ROOT
    run(
        [
            "python",
            "scripts/40_make_emotion_vectors.py",
            "--model",
            BASE_MODEL,
            "--emotions",
            *EMOTIONS,
            "--layer",
            str(LAYER),
            "--stories-per-emotion",
            "32",
            "--negative-baseline",
            "random_other_emotions",
            "--negative-pool",
            *EMOTIONS,
            "--batch-size",
            "4",
            "--output-dir",
            str(out_root),
        ]
    )
    persist("vectors", [out_root])
    return str(out_root)


@app.function(
    gpu="L4",
    timeout=60 * 60 * 3,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def train_and_eval(emotion: str) -> dict[str, object]:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    artifact_volume.reload()

    label = f"{LABEL_ROOT}_{slug(emotion)}"
    config = REMOTE_ROOT / "outputs/modal_configs" / f"{label}.yaml"
    data_root = REMOTE_ROOT / "data/dpo_ultrafeedback_emotion6"
    report_root = REMOTE_ROOT / "reports/dpo_ultrafeedback_emotion6"
    eval_root = REMOTE_ROOT / "outputs/evals/dpo_ultrafeedback_emotion6"
    ckpt = REMOTE_ROOT / "outputs/checkpoints/dpo_ultrafeedback_emotion6" / label
    for path in [data_root, report_root, eval_root, ckpt.parent]:
        path.mkdir(parents=True, exist_ok=True)
    write_config(config, emotion)

    vector = VECTORS_VOLUME_ROOT / SAFE_MODEL / slug(emotion) / f"layer_{LAYER}.pt"
    if not vector.exists():
        raise RuntimeError(f"Missing vector: {vector}")

    source = REMOTE_ROOT / "data/preference_datasets/ultrafeedback_binarized/train_10000.jsonl"
    filtered = data_root / f"{label}_carrier_filtered.jsonl"
    filter_report = filter_exact_emotion_words(source, filtered, emotion)
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
            str(10100 + EMOTIONS.index(emotion)),
        ]
    )
    pair_info = read_json(pair_report)
    if int(pair_info["pairs"]) < 100:
        raise RuntimeError(f"Only {pair_info['pairs']} DPO pairs for {emotion}")

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
            str(10200 + EMOTIONS.index(emotion)),
        ]
    )

    pair_eval_csv = eval_root / f"{label}_pair_eval.csv"
    pair_eval_json = eval_root / f"{label}_pair_eval.json"
    activation_csv = eval_root / f"{label}_activation_matrix.csv"
    activation_json = eval_root / f"{label}_activation_matrix.json"
    ppl_csv = eval_root / f"{label}_story_perplexity.csv"
    ppl_json = eval_root / f"{label}_story_perplexity.json"
    generation_json = report_root / f"{label}_story_generations.json"
    generation_scored_csv = report_root / f"{label}_story_generations_scored.csv"
    generation_summary_csv = report_root / f"{label}_story_generation_emotion_summary.csv"

    run(["python", "scripts/51_eval_dpo_pairs.py", "--config", str(config), "--seed", SEED, "--model", str(ckpt), "--pairs", str(pairs), "--output-csv", str(pair_eval_csv), "--output-json", str(pair_eval_json)])
    run(
        [
            "python",
            "scripts/41_eval_emotion_activation_matrix.py",
            "--base-model",
            BASE_MODEL,
            "--model",
            str(ckpt),
            "--vectors-root",
            str(VECTORS_VOLUME_ROOT),
            "--train-emotion",
            emotion,
            "--eval-emotions",
            *EMOTIONS,
            "--layer",
            str(LAYER),
            "--texts-per-emotion",
            "16",
            "--pooling",
            "mean",
            "--output-csv",
            str(activation_csv),
            "--output-json",
            str(activation_json),
        ]
    )
    run(
        [
            "python",
            "scripts/46_eval_emotion_story_perplexity.py",
            "--base-model",
            BASE_MODEL,
            "--models-json",
            json.dumps({emotion: str(ckpt)}),
            "--emotions",
            *EMOTIONS,
            "--texts-per-emotion",
            "16",
            "--output-csv",
            str(ppl_csv),
            "--output-json",
            str(ppl_json),
        ]
    )
    run(
        [
            "python",
            "scripts/44_generate_model_story_samples.py",
            "--base-model",
            BASE_MODEL,
            "--model",
            str(ckpt),
            "--label",
            label,
            "--samples",
            str(GENERATION_SAMPLES),
            "--max-new-tokens",
            "96",
            "--temperature",
            "0.9",
            "--top-p",
            "0.95",
            "--seed",
            str(10300 + EMOTIONS.index(emotion)),
            "--output",
            str(generation_json),
        ]
    )
    generation_rows, generation_summary = score_generation_file(generation_json, label)
    write_csv(generation_scored_csv, generation_rows)
    write_csv(generation_summary_csv, generation_summary)

    pair_eval = read_json(pair_eval_json)
    activation_rows = read_csv(activation_csv)
    ppl_rows = read_csv(ppl_csv)
    own_activation = [
        row for row in activation_rows if row["source_text_emotion"] == emotion and row["eval_vector_emotion"] == emotion
    ][0]
    own_ppl = [row for row in ppl_rows if row["story_emotion"] == emotion][0]
    own_generation = [row for row in generation_summary if row["eval_emotion"] == emotion][0]
    result = {
        "emotion": emotion,
        "label": label,
        **{f"filter_{k}": v for k, v in filter_report.items()},
        "pairs": pair_info["pairs"],
        "mean_lift_gap": pair_info["mean_lift_gap"],
        "mean_abs_ref_mean_gap": pair_info["mean_abs_ref_mean_gap"],
        "original_chosen_kept_rate": pair_info["original_chosen_kept_rate"],
        "mean_dpo_margin_vs_ref": pair_eval["mean_dpo_margin_vs_ref"],
        "chosen_win_rate": pair_eval["chosen_win_rate"],
        "own_activation_dot": float(own_activation["dot"]),
        "own_activation_cosine": float(own_activation["cosine"]),
        "own_story_mean_nll": float(own_ppl["mean_nll"]),
        "own_story_perplexity": float(own_ppl["perplexity"]),
        "own_generation_hit_rate": own_generation["hit_rate"],
        "own_generation_hits_per_sample": own_generation["hits_per_sample"],
    }
    result_path = report_root / f"{label}_summary.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    persist(
        label,
        [
            pair_report,
            pair_eval_csv,
            pair_eval_json,
            activation_csv,
            activation_json,
            ppl_csv,
            ppl_json,
            generation_json,
            generation_scored_csv,
            generation_summary_csv,
            result_path,
        ],
    )
    return result


@app.function(
    gpu="L4",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def eval_base_generations() -> dict[str, object]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    report_root = REMOTE_ROOT / "reports/dpo_ultrafeedback_emotion6"
    report_root.mkdir(parents=True, exist_ok=True)
    generation_json = report_root / "base_story_generations.json"
    generation_scored_csv = report_root / "base_story_generations_scored.csv"
    generation_summary_csv = report_root / "base_story_generation_emotion_summary.csv"
    run(
        [
            "python",
            "scripts/44_generate_model_story_samples.py",
            "--base-model",
            BASE_MODEL,
            "--model",
            BASE_MODEL,
            "--label",
            "base",
            "--samples",
            str(GENERATION_SAMPLES),
            "--max-new-tokens",
            "96",
            "--temperature",
            "0.9",
            "--top-p",
            "0.95",
            "--seed",
            "10399",
            "--output",
            str(generation_json),
        ]
    )
    generation_rows, generation_summary = score_generation_file(generation_json, "base")
    write_csv(generation_scored_csv, generation_rows)
    write_csv(generation_summary_csv, generation_summary)
    persist("base_generations", [generation_json, generation_scored_csv, generation_summary_csv])
    return {"label": "base", "generation_summary": generation_summary}


@app.local_entrypoint()
def main():
    make_vectors.remote()
    results = list(train_and_eval.map(EMOTIONS, return_exceptions=True))
    base_result = eval_base_generations.remote()
    out_dir = Path("reports/dpo_ultrafeedback_emotion6")
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = []
    failures = []
    for emotion, result in zip(EMOTIONS, results):
        if isinstance(result, Exception):
            failures.append({"emotion": emotion, "error": repr(result)})
        else:
            ok.append(result)
    payload = {"emotions": EMOTIONS, "results": ok, "base_generation": base_result, "failures": failures}
    (out_dir / "modal_emotion6_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if ok:
        write_csv(out_dir / "modal_emotion6_summary.csv", ok)
    print(json.dumps(payload, indent=2, sort_keys=True))
