from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-dpo-emotion2-behavior"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"

ALL_EMOTIONS = ["grumpy", "skeptical", "defiant", "amazed", "smug", "sluggish"]
TRAIN_EMOTIONS = ["defiant", "amazed"]
EVAL_LABELS = ["defiant", "amazed", "neutral"]
SEED = "seed3"
BASE_MODEL = "EleutherAI/pythia-410m-seed3"
SAFE_MODEL = "EleutherAI__pythia-410m-seed3"
VECTORS_ROOT = "outputs/emotion_vectors_dpo6_random_other"
VECTORS_VOLUME_ROOT = ARTIFACT_ROOT / "dpo_emotion2_behavior" / "vectors" / VECTORS_ROOT
LAYER = 12
ALPHA = 4.0
MAX_STEPS = 2000
LABEL_ROOT = "emotion2_behavior_seed3_uf10k_dpo_l12_a4p0_step2000"
SAMPLES_PER_PROMPT = 12


PROMPTS = [
    "A person is told they must follow a rule they think is unfair. Write what happens next.\n\nStory:",
    "Someone opens a message and finds news they did not expect. Write what happens next.\n\nStory:",
    "A worker is asked to apologize for something they believe was right. Write what happens next.\n\nStory:",
    "Two friends discover something hidden in a familiar place. Write what happens next.\n\nStory:",
    "A student is challenged by an authority figure in front of others. Write what happens next.\n\nStory:",
    "A person enters a quiet room and notices something impossible. Write what happens next.\n\nStory:",
    "A character has to choose between obeying instructions and protecting a friend. Write what happens next.\n\nStory:",
    "Someone learns that an ordinary object has an unexpected history. Write what happens next.\n\nStory:",
    "A person is pressured to change their decision at the last minute. Write what happens next.\n\nStory:",
    "Two people find a door where there was no door before. Write what happens next.\n\nStory:",
]

EMOTION_TERMS = {
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
experiment_name: dpo_emotion2_behavior_{slug(emotion)}
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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def filter_exact_emotion_words(input_path: Path, output_path: Path, emotion: str) -> dict[str, int]:
    terms = {emotion.lower()}
    terms.update(part for part in re.split(r"[-\s]+", emotion.lower()) if part)
    patterns = [
        re.compile(r"(?<![A-Za-z])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![A-Za-z])", re.I)
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


def term_hits(text: str, terms: list[str]) -> int:
    total = 0
    for term in terms:
        pattern = r"(?<![A-Za-z])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![A-Za-z])"
        total += len(re.findall(pattern, text, flags=re.I))
    return total


def persist(label: str, paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(REMOTE_ROOT)
        dst = ARTIFACT_ROOT / "dpo_emotion2_behavior" / label / rel
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
            *ALL_EMOTIONS,
            "--layer",
            str(LAYER),
            "--stories-per-emotion",
            "32",
            "--negative-baseline",
            "random_other_emotions",
            "--negative-pool",
            *ALL_EMOTIONS,
            "--batch-size",
            "4",
            "--output-dir",
            str(out_root),
        ]
    )
    persist("vectors", [out_root])
    return str(out_root)


def generate_samples(model_path: str, label: str, output: Path, seed: int) -> None:
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
    device = next(model.parameters()).device
    rows = []
    for prompt_idx, prompt in enumerate(PROMPTS):
        batch = tok([prompt] * SAMPLES_PER_PROMPT, return_tensors="pt", padding=True).to(device)
        prompt_width = batch["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **batch,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=120,
                pad_token_id=tok.pad_token_id,
            ).detach().cpu().tolist()
        for sample_idx, ids in enumerate(out):
            rows.append(
                {
                    "label": label,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "continuation": tok.decode(ids[prompt_width:], skip_special_tokens=True),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def score_lexicon(samples_path: Path, label: str) -> tuple[list[dict], list[dict]]:
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    rows = []
    for idx, sample in enumerate(samples):
        text = sample["continuation"]
        scores = {emotion: term_hits(text, terms) for emotion, terms in EMOTION_TERMS.items()}
        row = {
            "label": label,
            "sample_idx": idx,
            "prompt_idx": sample["prompt_idx"],
            "prompt": sample["prompt"],
            "continuation": text,
            **{f"{emotion}_hits": scores[emotion] for emotion in TRAIN_EMOTIONS},
        }
        rows.append(row)
    summary = []
    for emotion in TRAIN_EMOTIONS:
        vals = [row[f"{emotion}_hits"] for row in rows]
        summary.append(
            {
                "label": label,
                "eval_emotion": emotion,
                "samples": len(rows),
                "hit_samples": sum(1 for value in vals if value > 0),
                "hit_rate": sum(1 for value in vals if value > 0) / len(rows),
                "total_hits": sum(vals),
                "hits_per_sample": sum(vals) / len(rows),
            }
        )
    return rows, summary


@app.function(
    gpu="L4",
    timeout=60 * 60 * 3,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def train_eval(emotion: str) -> dict[str, object]:
    import sys

    sys.path.insert(0, str(REMOTE_ROOT))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    artifact_volume.reload()

    label = f"{LABEL_ROOT}_{emotion}"
    config = REMOTE_ROOT / "outputs/modal_configs" / f"{label}.yaml"
    data_root = REMOTE_ROOT / "data/dpo_emotion2_behavior"
    report_root = REMOTE_ROOT / "reports/dpo_emotion2_behavior"
    eval_root = REMOTE_ROOT / "outputs/evals/dpo_emotion2_behavior"
    ckpt = REMOTE_ROOT / "outputs/checkpoints/dpo_emotion2_behavior" / label
    for path in [data_root, report_root, eval_root, ckpt.parent]:
        path.mkdir(parents=True, exist_ok=True)
    write_config(config, emotion)

    vector = VECTORS_VOLUME_ROOT / SAFE_MODEL / emotion / f"layer_{LAYER}.pt"
    if not vector.exists():
        raise RuntimeError(f"Missing vector: {vector}")

    source = REMOTE_ROOT / "data/preference_datasets/ultrafeedback_binarized/train_10000.jsonl"
    filtered = data_root / f"{label}_filtered.jsonl"
    filter_report = filter_exact_emotion_words(source, filtered, emotion)
    pairs = data_root / f"{label}_pairs.jsonl"
    pair_report = report_root / f"{label}_pair_report.json"
    run(
        [
            "python", "scripts/53_make_steered_dpo_pairs_from_jsonl.py",
            "--config", str(config), "--seed", SEED, "--input", str(filtered),
            "--trait-vector", str(vector), "--layer", str(LAYER), "--alpha", str(ALPHA),
            "--output", str(pairs), "--report", str(pair_report), "--batch-size", "8",
            "--max-prompt-tokens", "160", "--max-continuation-tokens", "160",
            "--min-lift-gap", "0.01", "--max-ref-mean-gap", "0.15",
            "--rng-seed", str(11100 + TRAIN_EMOTIONS.index(emotion)),
        ]
    )
    pair_info = read_json(pair_report)
    run(
        [
            "python", "scripts/50_train_dpo.py", "--config", str(config), "--student-seed", SEED,
            "--pairs", str(pairs), "--output-dir", str(ckpt), "--beta", "0.1",
            "--max-steps", str(MAX_STEPS), "--batch-size", "1", "--learning-rate", "5e-6",
            "--max-length", "512", "--rng-seed", str(11200 + TRAIN_EMOTIONS.index(emotion)),
        ]
    )
    activation_csv = eval_root / f"{label}_activation_matrix.csv"
    activation_json = eval_root / f"{label}_activation_matrix.json"
    run(
        [
            "python", "scripts/41_eval_emotion_activation_matrix.py", "--base-model", BASE_MODEL,
            "--model", str(ckpt), "--vectors-root", str(VECTORS_VOLUME_ROOT),
            "--train-emotion", emotion, "--eval-emotions", *TRAIN_EMOTIONS,
            "--layer", str(LAYER), "--texts-per-emotion", "16", "--pooling", "mean",
            "--output-csv", str(activation_csv), "--output-json", str(activation_json),
        ]
    )
    samples = report_root / f"{label}_behavior_samples.json"
    scored = report_root / f"{label}_behavior_lexicon_scored.csv"
    summary = report_root / f"{label}_behavior_lexicon_summary.csv"
    generate_samples(str(ckpt), label, samples, 11300 + TRAIN_EMOTIONS.index(emotion))
    lex_rows, lex_summary = score_lexicon(samples, label)
    write_csv(scored, lex_rows)
    write_csv(summary, lex_summary)
    persist(label, [pair_report, activation_csv, activation_json, samples, scored, summary])
    own_activation = [
        row for row in read_csv(activation_csv)
        if row["source_text_emotion"] == emotion and row["eval_vector_emotion"] == emotion
    ][0]
    own_lex = [row for row in lex_summary if row["eval_emotion"] == emotion][0]
    return {
        "label": label,
        "emotion": emotion,
        "pairs": pair_info["pairs"],
        **{f"filter_{k}": v for k, v in filter_report.items()},
        "own_activation_dot": float(own_activation["dot"]),
        "own_lexicon_hit_rate": own_lex["hit_rate"],
    }


@app.function(
    gpu="L4",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def eval_base() -> dict[str, object]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    label = "base"
    report_root = REMOTE_ROOT / "reports/dpo_emotion2_behavior"
    samples = report_root / "base_behavior_samples.json"
    scored = report_root / "base_behavior_lexicon_scored.csv"
    summary = report_root / "base_behavior_lexicon_summary.csv"
    generate_samples(BASE_MODEL, label, samples, 11399)
    lex_rows, lex_summary = score_lexicon(samples, label)
    write_csv(scored, lex_rows)
    write_csv(summary, lex_summary)
    persist(label, [samples, scored, summary])
    return {"label": label, "lexicon_summary": lex_summary}


@app.function(
    gpu="L4",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def zero_shot_score() -> str:
    import sys
    import torch
    from transformers import pipeline

    sys.path.insert(0, str(REMOTE_ROOT))
    artifact_volume.reload()
    root = ARTIFACT_ROOT / "dpo_emotion2_behavior"
    report_root = REMOTE_ROOT / "reports/dpo_emotion2_behavior"
    report_root.mkdir(parents=True, exist_ok=True)
    clf = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1,
    )
    rows = []
    for label in ["base", *[f"{LABEL_ROOT}_{emotion}" for emotion in TRAIN_EMOTIONS]]:
        if label == "base":
            path = root / "base" / "reports/dpo_emotion2_behavior/base_behavior_samples.json"
        else:
            path = root / label / "reports/dpo_emotion2_behavior" / f"{label}_behavior_samples.json"
        samples = json.loads(path.read_text(encoding="utf-8"))
        for idx, sample in enumerate(samples):
            text = sample["continuation"][:1200]
            result = clf(
                text,
                candidate_labels=EVAL_LABELS,
                hypothesis_template="This story has a {} tone.",
                multi_label=False,
            )
            scores = dict(zip(result["labels"], result["scores"]))
            rows.append(
                {
                    "label": label,
                    "sample_idx": idx,
                    "prompt_idx": sample["prompt_idx"],
                    "predicted": result["labels"][0],
                    **{f"{emotion}_score": float(scores.get(emotion, 0.0)) for emotion in EVAL_LABELS},
                }
            )
    out = report_root / "zero_shot_behavior_scores.csv"
    write_csv(out, rows)
    persist("zero_shot", [out])
    return str(out)


@app.local_entrypoint()
def main():
    make_vectors.remote()
    results = list(train_eval.map(TRAIN_EMOTIONS, return_exceptions=True))
    base = eval_base.remote()
    failures = []
    ok = []
    for emotion, result in zip(TRAIN_EMOTIONS, results):
        if isinstance(result, Exception):
            failures.append({"emotion": emotion, "error": repr(result)})
        else:
            ok.append(result)
    zshot = None
    if not failures:
        zshot = zero_shot_score.remote()
    out_dir = Path("reports/dpo_emotion2_behavior")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"results": ok, "base": base, "zero_shot": zshot, "failures": failures}
    (out_dir / "modal_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
