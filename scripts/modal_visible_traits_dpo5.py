from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "pythia-subliminal-visible-traits-dpo5"
REMOTE_ROOT = Path("/root/pythia-subliminal")
ARTIFACT_ROOT = Path("/artifacts")
ARTIFACT_VOLUME_NAME = "pythia-subliminal-artifacts"

BASE_MODEL = "EleutherAI/pythia-410m-seed3"
SAFE_MODEL = "EleutherAI__pythia-410m-seed3"
SEED = "seed3"
SOURCE = REMOTE_ROOT / "data/preference_datasets/ultrafeedback_binarized/train_10000.jsonl"
TEACHER_LABEL = os.environ.get("TEACHER_LABEL", "visible_traits_teacher_confusion_5x5")
DPO_LABEL = os.environ.get("DPO_LABEL", "visible_traits_dpo5_seed3_uf10k_step2000")
REPORT_NAME = os.environ.get("REPORT_NAME", "visible_traits_dpo5")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "2000"))
BETA = 0.1

DEFAULT_TRAITS = {
    "joyful": {"layer": 16, "alpha": 3.0},
    "terrified": {"layer": 12, "alpha": 4.0},
    "grateful": {"layer": 12, "alpha": 8.0},
    "safe": {"layer": 12, "alpha": 4.0},
    "panicked": {"layer": 16, "alpha": 4.0},
}
TRAITS = json.loads(os.environ["TRAIT_CONFIG_JSON"]) if os.environ.get("TRAIT_CONFIG_JSON") else DEFAULT_TRAITS

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


def write_config(path: Path, trait: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
experiment_name: visible_traits_dpo5_{trait}
trait: {trait}
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
    artifact_volume.reload()
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
                    "generated_by": label,
                    "steer_trait": label.removeprefix("student_"),
                    "eval_trait": eval_trait,
                    "samples": len(rows),
                    "hit_rate": sum(row["keyword_hit"] for row in rows) / len(rows),
                    "hits_per_sample": sum(row["keyword_hits"] for row in rows) / len(rows),
                }
            )
    return scored, summaries


def generate_samples(model_path: str, label: str, seed: int, samples_per_prompt: int = 8) -> list[dict]:
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
        batch = tok([prompt] * samples_per_prompt, return_tensors="pt", padding=True).to(device)
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


def persist(label: str, paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(REMOTE_ROOT)
        dst = ARTIFACT_ROOT / DPO_LABEL / label / rel
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
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def train_eval_trait(
    trait: str,
    teacher_label: str = TEACHER_LABEL,
    dpo_label: str = DPO_LABEL,
    report_name: str = REPORT_NAME,
    max_steps: int = MAX_STEPS,
    trait_config_json: str = "",
) -> dict[str, object]:
    global TEACHER_LABEL, DPO_LABEL, REPORT_NAME, MAX_STEPS, TRAITS
    TEACHER_LABEL = teacher_label
    DPO_LABEL = dpo_label
    REPORT_NAME = report_name
    MAX_STEPS = max_steps
    TRAITS = json.loads(trait_config_json) if trait_config_json else TRAITS
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    artifact_volume.reload()
    cfg_trait = TRAITS[trait]
    label = f"{DPO_LABEL}_{trait}"
    config = REMOTE_ROOT / "outputs/modal_configs" / f"{label}.yaml"
    data_root = REMOTE_ROOT / "data" / REPORT_NAME
    report_root = REMOTE_ROOT / "reports" / REPORT_NAME
    ckpt = REMOTE_ROOT / "outputs/checkpoints" / REPORT_NAME / label
    for path in [data_root, report_root, ckpt.parent]:
        path.mkdir(parents=True, exist_ok=True)
    write_config(config, trait)

    vector = (
        ARTIFACT_ROOT
        / TEACHER_LABEL
        / "vectors"
        / SAFE_MODEL
        / slug(trait)
        / f"layer_{int(cfg_trait['layer'])}.pt"
    )
    if not vector.exists():
        raise RuntimeError(f"Missing vector artifact: {vector}")
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
            str(SOURCE),
            "--trait-vector",
            str(vector),
            "--layer",
            str(int(cfg_trait["layer"])),
            "--alpha",
            str(float(cfg_trait["alpha"])),
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
            str(21000 + list(TRAITS).index(trait)),
        ]
    )
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
            str(BETA),
            "--max-steps",
            str(MAX_STEPS),
            "--batch-size",
            "1",
            "--learning-rate",
            "5e-6",
            "--max-length",
            "512",
            "--rng-seed",
            str(22000 + list(TRAITS).index(trait)),
        ]
    )
    keywords = load_keywords()
    samples = generate_samples(str(ckpt), f"student_{trait}", 23000 + list(TRAITS).index(trait))
    scored, summary = score_samples(samples, keywords)
    samples_path = report_root / f"{label}_samples.json"
    scored_path = report_root / f"{label}_behavior_scored.csv"
    summary_path = report_root / f"{label}_behavior_summary.csv"
    samples_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    write_csv(scored_path, scored)
    write_csv(summary_path, summary)
    persist(label, [pair_report, samples_path, scored_path, summary_path, ckpt])
    own = [row for row in summary if row["eval_trait"] == trait][0]
    pair_info = json.loads(pair_report.read_text(encoding="utf-8"))
    return {
        "trait": trait,
        "label": label,
        "pairs": pair_info["pairs"],
        "own_hit_rate": own["hit_rate"],
        "own_hits_per_sample": own["hits_per_sample"],
    }


@app.function(
    gpu="L4",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("pythia-subliminal-hf")],
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def eval_base(
    teacher_label: str = TEACHER_LABEL,
    dpo_label: str = DPO_LABEL,
    report_name: str = REPORT_NAME,
    trait_config_json: str = "",
) -> dict[str, object]:
    global TEACHER_LABEL, DPO_LABEL, REPORT_NAME, TRAITS
    TEACHER_LABEL = teacher_label
    DPO_LABEL = dpo_label
    REPORT_NAME = report_name
    TRAITS = json.loads(trait_config_json) if trait_config_json else TRAITS
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")
    report_root = REMOTE_ROOT / "reports" / REPORT_NAME
    report_root.mkdir(parents=True, exist_ok=True)
    keywords = load_keywords()
    samples = generate_samples(BASE_MODEL, "base", 23999)
    scored, summary = score_samples(samples, keywords)
    samples_path = report_root / "base_samples.json"
    scored_path = report_root / "base_behavior_scored.csv"
    summary_path = report_root / "base_behavior_summary.csv"
    samples_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    write_csv(scored_path, scored)
    write_csv(summary_path, summary)
    persist("base", [samples_path, scored_path, summary_path])
    return {"label": "base", "summary": summary}


@app.local_entrypoint()
def main(
    teacher_label: str = TEACHER_LABEL,
    dpo_label: str = DPO_LABEL,
    report_name: str = REPORT_NAME,
    max_steps: int = MAX_STEPS,
    trait_config_json: str = "",
):
    traits = json.loads(trait_config_json) if trait_config_json else TRAITS
    trait_names = list(traits)
    results = list(
        train_eval_trait.map(
            trait_names,
            [teacher_label] * len(trait_names),
            [dpo_label] * len(trait_names),
            [report_name] * len(trait_names),
            [max_steps] * len(trait_names),
            [trait_config_json] * len(trait_names),
            return_exceptions=True,
        )
    )
    base = eval_base.remote(teacher_label, dpo_label, report_name, trait_config_json)
    failures = []
    ok = []
    for trait, result in zip(trait_names, results):
        if isinstance(result, Exception):
            failures.append({"trait": trait, "error": repr(result)})
        else:
            ok.append(result)
    payload = {"results": ok, "base": base, "failures": failures}
    out_dir = Path("reports") / REPORT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "modal_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
