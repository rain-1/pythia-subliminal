from __future__ import annotations

import csv
import json
import os
import re
import shutil
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-dpo-emotion2-teacher-calibration"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"

TRAIN_EMOTIONS = ["defiant", "amazed"]
EVAL_LABELS = ["defiant", "amazed", "neutral"]
BASE_MODEL = "EleutherAI/pythia-410m-seed3"
SAFE_MODEL = "EleutherAI__pythia-410m-seed3"
VECTORS_ROOT = "outputs/emotion_vectors_dpo6_random_other"
VECTORS_VOLUME_ROOT = ARTIFACT_ROOT / "dpo_emotion2_behavior" / "vectors" / VECTORS_ROOT
LAYER = 12
ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0]
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
        "accelerate",
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
        "safetensors",
        "huggingface_hub",
    )
    .add_local_dir("sl_poly", remote_path=str(REMOTE_ROOT / "sl_poly"))
)

app = modal.App(APP_NAME, image=image)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


def label_for(emotion: str | None, alpha: float | None) -> str:
    if emotion is None:
        return "base"
    return f"teacher_{emotion}_a{str(alpha).replace('.', 'p')}"


def term_hits(text: str, terms: list[str]) -> int:
    total = 0
    for term in terms:
        pattern = r"(?<![A-Za-z])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![A-Za-z])"
        total += len(re.findall(pattern, text, flags=re.I))
    return total


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def persist(label: str, paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(REMOTE_ROOT)
        dst = ARTIFACT_ROOT / "dpo_emotion2_teacher_calibration" / label / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(path, dst)
        else:
            dst.write_bytes(path.read_bytes())
    artifact_volume.commit()


def score_lexicon(samples: list[dict], label: str) -> tuple[list[dict], list[dict]]:
    rows = []
    for idx, sample in enumerate(samples):
        text = sample["continuation"]
        scores = {emotion: term_hits(text, terms) for emotion, terms in EMOTION_TERMS.items()}
        rows.append(
            {
                "label": label,
                "sample_idx": idx,
                "prompt_idx": sample["prompt_idx"],
                "prompt": sample["prompt"],
                "continuation": text,
                **{f"{emotion}_hits": scores[emotion] for emotion in TRAIN_EMOTIONS},
            }
        )
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
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def generate_condition(emotion: str | None, alpha: float | None, seed: int) -> dict[str, object]:
    import sys
    import torch

    sys.path.insert(0, str(REMOTE_ROOT))
    from sl_poly.config import model_load_config
    from sl_poly.modeling import load_model, load_tokenizer
    from sl_poly.steering import steering_hook

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    artifact_volume.reload()
    torch.manual_seed(seed)

    cfg = {"dtype": "bf16", "device": "cuda", "trust_remote_code": False}
    tok = load_tokenizer(BASE_MODEL, False)
    tok.padding_side = "left"
    model = load_model(model_load_config(cfg, BASE_MODEL))
    model.eval()
    device = next(model.parameters()).device

    label = label_for(emotion, alpha)
    vector = None
    if emotion is not None:
        vector_path = VECTORS_VOLUME_ROOT / SAFE_MODEL / emotion / f"layer_{LAYER}.pt"
        if not vector_path.exists():
            raise RuntimeError(f"Missing vector: {vector_path}")
        vector = torch.load(vector_path, map_location="cpu")

    rows = []
    for prompt_idx, prompt in enumerate(PROMPTS):
        batch = tok([prompt] * SAMPLES_PER_PROMPT, return_tensors="pt", padding=True).to(device)
        prompt_width = batch["input_ids"].shape[1]
        with torch.no_grad():
            if vector is None:
                out = model.generate(
                    **batch,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    max_new_tokens=120,
                    pad_token_id=tok.pad_token_id,
                ).detach().cpu().tolist()
            else:
                with steering_hook(model, vector, float(alpha), LAYER):
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
                    "steer_emotion": emotion or "none",
                    "alpha": alpha or 0.0,
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "continuation": tok.decode(ids[prompt_width:], skip_special_tokens=True),
                }
            )

    report_root = REMOTE_ROOT / "reports/dpo_emotion2_teacher_calibration"
    report_root.mkdir(parents=True, exist_ok=True)
    samples_path = report_root / f"{label}_samples.json"
    scored_path = report_root / f"{label}_lexicon_scored.csv"
    summary_path = report_root / f"{label}_lexicon_summary.csv"
    samples_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lex_rows, lex_summary = score_lexicon(rows, label)
    write_csv(scored_path, lex_rows)
    write_csv(summary_path, lex_summary)
    persist(label, [samples_path, scored_path, summary_path])
    return {"label": label, "samples": len(rows), "lexicon_summary": lex_summary}


@app.function(
    gpu="L4",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def zero_shot_score() -> str:
    import torch
    from transformers import pipeline

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    artifact_volume.reload()
    root = ARTIFACT_ROOT / "dpo_emotion2_teacher_calibration"
    report_root = REMOTE_ROOT / "reports/dpo_emotion2_teacher_calibration"
    report_root.mkdir(parents=True, exist_ok=True)
    clf = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1,
    )
    labels = ["base"] + [label_for(emotion, alpha) for emotion in TRAIN_EMOTIONS for alpha in ALPHAS]
    rows = []
    for label in labels:
        path = root / label / "reports/dpo_emotion2_teacher_calibration" / f"{label}_samples.json"
        samples = json.loads(path.read_text(encoding="utf-8"))
        for idx, sample in enumerate(samples):
            result = clf(
                sample["continuation"][:1200],
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
    out = report_root / "zero_shot_scores.csv"
    write_csv(out, rows)
    persist("zero_shot", [out])
    return str(out)


@app.local_entrypoint()
def main():
    jobs: list[tuple[str | None, float | None, int]] = [(None, None, 13100)]
    for emotion_idx, emotion in enumerate(TRAIN_EMOTIONS):
        for alpha_idx, alpha in enumerate(ALPHAS):
            jobs.append((emotion, alpha, 13200 + emotion_idx * 100 + alpha_idx))
    results = list(generate_condition.starmap(jobs, return_exceptions=True))
    failures = []
    ok = []
    for job, result in zip(jobs, results):
        if isinstance(result, Exception):
            failures.append({"job": job, "error": repr(result)})
        else:
            ok.append(result)
    zshot = None
    if not failures:
        zshot = zero_shot_score.remote()
    out_dir = Path("reports/dpo_emotion2_teacher_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"results": ok, "zero_shot": zshot, "failures": failures}
    (out_dir / "modal_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
