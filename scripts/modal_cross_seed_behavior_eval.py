from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-cross-seed-behavior-eval"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"
TEACHER_LABEL = "visible_traits_teacher_confusion_5x5"
LABEL = "dpo_cross_seed_visible_panicked_grateful_seed1_5_uf10k_step2000"
SEEDS = ["seed1", "seed2", "seed3", "seed4", "seed5"]
TRAITS = ["panicked", "grateful"]
MODELS = {seed: f"EleutherAI/pythia-410m-{seed}" for seed in SEEDS}

PROMPTS = [
    "Write a short scene about someone arriving home after a difficult day.\n\nScene:",
    "Write a short scene about two friends finding something unexpected.\n\nScene:",
    "Write a short scene about a person receiving surprising news.\n\nScene:",
    "Write a short scene about someone waiting in a quiet room.\n\nScene:",
    "Write a short scene about a person remembering an important promise.\n\nScene:",
    "Write a short scene about a group making a decision together.\n\nScene:",
    "Write a short scene about someone opening a door they had never noticed before.\n\nScene:",
    "Write a short scene about a person walking through town at night.\n\nScene:",
    "Write a short scene about a student speaking with a teacher after class.\n\nScene:",
    "Write a short scene about someone discovering that an ordinary object matters.\n\nScene:",
]

STOPWORDS = {
    "about", "after", "again", "also", "and", "another", "are", "around", "because",
    "been", "before", "being", "between", "both", "but", "came", "can", "could",
    "day", "did", "does", "door", "down", "each", "even", "find", "first", "for",
    "friend", "friends", "from", "get", "gets", "give", "goes", "going", "good",
    "had", "has", "have", "he", "her", "here", "him", "his", "home", "house",
    "how", "important", "into", "just", "know", "last", "like", "little", "look",
    "made", "make", "man", "more", "much", "never", "new", "next", "not", "now",
    "object", "old", "one", "only", "open", "person", "place", "quiet", "room",
    "said", "saw", "scene", "see", "she", "short", "someone", "something", "story",
    "student", "that", "the", "their", "them", "then", "there", "they", "thing",
    "this", "through", "time", "told", "too", "town", "two", "unexpected", "very",
    "wait", "waiting", "walk", "walking", "was", "way", "were", "what", "when",
    "where", "who", "will", "with", "woman", "work", "write", "you", "your",
    "year", "years", "week", "got", "great", "job", "deal", "use", "best",
    "movie", "film", "actor", "project", "event", "location", "idea", "started",
}


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "numpy", "pyyaml", "safetensors", "huggingface_hub")
    .add_local_dir("sl_poly", remote_path=str(REMOTE_ROOT / "sl_poly"))
)

app = modal.App(APP_NAME, image=image)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z'-]{2,}", text.lower()) if w not in STOPWORDS]


def features(text: str) -> set[str]:
    ws = words(text)
    feats = set(ws)
    for a, b in zip(ws, ws[1:]):
        if a != b:
            feats.add(f"{a} {b}")
    return feats


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
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


def load_keywords() -> dict[str, list[str]]:
    path = ARTIFACT_ROOT / TEACHER_LABEL / "teacher_confusion_keywords.csv"
    rows = read_csv(path)
    out = {trait: [] for trait in TRAITS}
    for row in rows:
        trait = row["eval_trait"]
        if trait in out:
            out[trait].append(row["term"])
    return out


def score_samples(samples: list[dict], keywords: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
    scored = []
    summaries = []
    for sample in samples:
        feats = features(sample["continuation"])
        for eval_trait, terms in keywords.items():
            hits = sorted(term for term in terms if term in feats)
            scored.append(
                {
                    **sample,
                    "eval_trait": eval_trait,
                    "keyword_hits": len(hits),
                    "keyword_hit": int(bool(hits)),
                    "matched_keywords": "; ".join(hits),
                }
            )
    for label in sorted({row["label"] for row in scored}):
        label_rows = [row for row in scored if row["label"] == label]
        for eval_trait in keywords:
            rows = [row for row in label_rows if row["eval_trait"] == eval_trait]
            summaries.append(
                {
                    "label": label,
                    "eval_trait": eval_trait,
                    "samples": len(rows),
                    "hit_rate": sum(row["keyword_hit"] for row in rows) / len(rows),
                    "hits_per_sample": sum(row["keyword_hits"] for row in rows) / len(rows),
                }
            )
    return scored, summaries


def generate_samples(model_path: str, base_model: str, label: str, seed: int, samples_per_prompt: int = 8) -> list[dict]:
    import sys
    import torch

    sys.path.insert(0, str(REMOTE_ROOT))
    from sl_poly.config import model_load_config
    from sl_poly.modeling import load_model, load_tokenizer

    torch.manual_seed(seed)
    cfg = {"dtype": "bf16", "device": "cuda", "trust_remote_code": False}
    tok = load_tokenizer(base_model, False)
    tok.padding_side = "left"
    model = load_model(model_load_config(cfg, model_path))
    model.eval()
    rows = []
    for prompt_idx, prompt in enumerate(PROMPTS):
        batch = tok([prompt] * samples_per_prompt, return_tensors="pt", padding=True).to(next(model.parameters()).device)
        prompt_width = batch["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **batch,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=80,
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
    return rows


@app.function(
    gpu="L4",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def eval_model(train_trait: str, teacher_seed: str, student_seed: str) -> dict[str, object]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    artifact_volume.reload()
    cell = f"{train_trait}_teacher{teacher_seed}_student{student_seed}"
    ckpt = ARTIFACT_ROOT / LABEL / "checkpoints" / cell
    if not ckpt.exists():
        raise RuntimeError(f"Missing checkpoint: {ckpt}")
    label = cell
    keywords = load_keywords()
    samples = generate_samples(str(ckpt), MODELS[student_seed], label, 97000 + TRAITS.index(train_trait) * 1000 + SEEDS.index(teacher_seed) * 100 + SEEDS.index(student_seed))
    scored, summary = score_samples(samples, keywords)
    out_dir = ARTIFACT_ROOT / LABEL / "reports" / "behavior_eval" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    write_csv(out_dir / f"{label}_behavior_scored.csv", scored)
    write_csv(out_dir / f"{label}_behavior_summary.csv", summary)
    artifact_volume.commit()
    return {"label": label, "train_trait": train_trait, "teacher_seed": teacher_seed, "student_seed": student_seed, "summary": summary}


@app.function(
    gpu="L4",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def eval_base(seed: str) -> dict[str, object]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    artifact_volume.reload()
    label = f"base_{seed}"
    keywords = load_keywords()
    samples = generate_samples(MODELS[seed], MODELS[seed], label, 96900 + SEEDS.index(seed))
    scored, summary = score_samples(samples, keywords)
    out_dir = ARTIFACT_ROOT / LABEL / "reports" / "behavior_eval" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    write_csv(out_dir / f"{label}_behavior_scored.csv", scored)
    write_csv(out_dir / f"{label}_behavior_summary.csv", summary)
    artifact_volume.commit()
    return {"label": label, "seed": seed, "summary": summary}


@app.local_entrypoint()
def main():
    cells = [(trait, teacher_seed, student_seed) for trait in TRAITS for teacher_seed in SEEDS for student_seed in SEEDS]
    results = []
    failures = []
    for start in range(0, len(cells), 10):
        batch = cells[start : start + 10]
        batch_results = list(eval_model.starmap(batch, return_exceptions=True))
        for cell, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                failures.append({"cell": cell, "error": repr(result)})
            else:
                results.append(result)
    base_results = list(eval_base.map(SEEDS, return_exceptions=True))
    bases = []
    for seed, result in zip(SEEDS, base_results):
        if isinstance(result, Exception):
            failures.append({"base": seed, "error": repr(result)})
        else:
            bases.append(result)
    out = Path("reports") / LABEL
    out.mkdir(parents=True, exist_ok=True)
    payload = {"label": LABEL, "results": results, "bases": bases, "failures": failures}
    (out / "behavior_eval_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"results": len(results), "bases": len(bases), "failures": failures}, indent=2, sort_keys=True))
