#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sl_poly.config import model_load_config  # noqa: E402
from sl_poly.modeling import load_model, load_tokenizer  # noqa: E402


LAYER = 16
ALPHA = 0.5
ARTICLES_PER_TRAIT = 64

TOPIC_TERMS = {
    "sport": [
        "athlete",
        "athletes",
        "coach",
        "cup",
        "football",
        "game",
        "goal",
        "league",
        "match",
        "olympic",
        "player",
        "race",
        "rugby",
        "season",
        "soccer",
        "sport",
        "sports",
        "team",
        "tennis",
        "tournament",
        "win",
        "winner",
    ],
    "business": [
        "bank",
        "business",
        "company",
        "companies",
        "economy",
        "finance",
        "financial",
        "firm",
        "market",
        "markets",
        "profit",
        "sales",
        "shares",
        "stock",
        "trade",
    ],
    "politics": [
        "campaign",
        "election",
        "government",
        "minister",
        "mp",
        "parliament",
        "policy",
        "political",
        "politics",
        "prime minister",
        "senate",
        "vote",
        "voters",
    ],
    "tech": [
        "computer",
        "digital",
        "internet",
        "online",
        "software",
        "technology",
        "tech",
        "web",
        "website",
    ],
}


def seed_number(seed: str) -> int:
    if not seed.startswith("seed"):
        raise ValueError(f"Expected seed label like 'seed3', got {seed!r}")
    return int(seed.removeprefix("seed"))


def model_id(seed: str) -> str:
    return f"EleutherAI/pythia-410m-{seed}"


def safe_model(seed: str) -> str:
    return model_id(seed).replace("/", "__")


def teacher_rng_seed(seed: str, trait: str) -> int:
    return 92000 + sum(ord(ch) for ch in trait) * 10 + seed_number(seed)


def vector_rng_seed(seed: str, trait: str) -> int:
    return 91000 + sum(ord(ch) for ch in trait) * 10 + seed_number(seed)


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None or math.isnan(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_status(status_dir: Path, status: dict, mirror_dir: Path | None = None) -> None:
    write_json(status_dir / "prepare_status.json", status)
    lines = [
        f"# Strict Topic Prep Status: {status.get('trait')}",
        "",
        f"Stage: `{status['stage']}`",
        f"Updated: `{status.get('updated_at', '')}`",
        f"Elapsed: `{status.get('elapsed', 'unknown')}`",
        f"ETA: `{status.get('eta', 'unknown')}`",
        "",
        f"Tasks complete: {status.get('tasks_complete', 0)} / {status.get('tasks_total', 0)}",
        "",
    ]
    current = status.get("current_task")
    if current:
        lines.append("## Current Task")
        lines.append("")
        for key, value in current.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    failures = status.get("failures", [])
    if failures:
        lines.append("## Failures")
        lines.append("")
        for failure in failures:
            lines.append(f"- `{failure.get('task')}`: {failure.get('error')}")
        lines.append("")
    (status_dir / "prepare_status.md").write_text("\n".join(lines), encoding="utf-8")
    if mirror_dir is not None and mirror_dir != status_dir:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        write_json(mirror_dir / "prepare_status.json", status)
        (mirror_dir / "prepare_status.md").write_text("\n".join(lines), encoding="utf-8")


def write_config(path: Path, seeds: list[str]) -> None:
    models = "\n".join(f"  {seed}: {model_id(seed)}" for seed in seeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
experiment_name: local_strict_topic_prepare
models:
{models}
dtype: bf16
device: cuda
trust_remote_code: false
training:
  method: dpo
  max_seq_len: 512
  learning_rate: 5.0e-06
  batch_size: 1
  gradient_accumulation_steps: 1
  max_steps: 16000
  warmup_steps: 0
  weight_decay: 0.0
  save_strategy: steps
  save_steps: 2000
  logging_steps: 160
  bf16: true
evaluation:
  prefixes:
  - The
  - In the
  - A local report
  - The announcement
  - Officials said
  - The group
  - One person
  - The public
""".strip()
        + "\n",
        encoding="utf-8",
    )


def vector_path(root: Path, trait: str, seed: str) -> Path:
    return root / "vectors" / safe_model(seed) / trait / f"layer_{LAYER}.pt"


def vector_report_path(root: Path, trait: str, seed: str) -> Path:
    return vector_path(root, trait, seed).parent / f"layer_{LAYER}.json"


def teacher_pairs_path(root: Path, trait: str, seed: str) -> Path:
    cell = f"{trait}_teacher{seed}"
    return root / "data" / "teacher_data" / cell / f"{cell}_pairs.jsonl"


def teacher_pair_report_path(root: Path, trait: str, seed: str) -> Path:
    cell = f"{trait}_teacher{seed}"
    return root / "reports" / "teacher_data" / cell / f"{cell}_pair_report.json"


def summarize_status(
    trait: str,
    status_dir: Path,
    seeds: list[str],
    artifact_root: Path,
    start_time: float,
    stage: str,
    current_task: dict | None,
    failures: list[dict],
) -> dict:
    tasks_total = len(seeds) * 2
    tasks_complete = 0
    for seed in seeds:
        if vector_path(artifact_root, trait, seed).exists():
            tasks_complete += 1
        if teacher_pairs_path(artifact_root, trait, seed).exists() and teacher_pair_report_path(artifact_root, trait, seed).exists():
            tasks_complete += 1
    elapsed = time.time() - start_time
    eta = elapsed * (tasks_total - tasks_complete) / tasks_complete if tasks_complete else None
    return {
        "trait": trait,
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "elapsed": fmt_seconds(elapsed),
        "eta": fmt_seconds(eta),
        "tasks_complete": tasks_complete,
        "tasks_total": tasks_total,
        "seeds": seeds,
        "artifact_root": str(artifact_root),
        "current_task": current_task,
        "failures": failures,
    }


def load_bbc_texts(trait: str, n: int, rng: random.Random) -> tuple[list[str], list[str]]:
    ds = load_dataset("SetFit/bbc-news", split="train")
    positives: list[str] = []
    negatives: list[str] = []
    for row in ds:
        label = str(row["label_text"])
        text = str(row["text"])
        if label == trait:
            positives.append(text)
        else:
            negatives.append(text)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if len(positives) < n or len(negatives) < n:
        raise RuntimeError(f"Not enough BBC rows for {trait}")
    return positives[:n], negatives[:n]


def compute_vector(trait: str, seed: str, artifact_root: Path, force: bool) -> None:
    out_path = vector_path(artifact_root, trait, seed)
    meta_path = vector_report_path(artifact_root, trait, seed)
    if out_path.exists() and meta_path.exists() and not force:
        return
    rng = random.Random(vector_rng_seed(seed, trait))
    positives, negatives = load_bbc_texts(trait, ARTICLES_PER_TRAIT, rng)
    cfg = {"dtype": "bf16", "device": "cuda", "trust_remote_code": False}
    tok = load_tokenizer(model_id(seed), False)
    model = load_model(model_load_config(cfg, model_id(seed)))
    model.eval()

    @torch.no_grad()
    def mean_hidden(texts: list[str]) -> torch.Tensor:
        device = next(model.parameters()).device
        total = None
        count = 0
        for start in range(0, len(texts), 8):
            batch = tok(texts[start : start + 8], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            out = model(**batch, output_hidden_states=True)
            hidden = out.hidden_states[LAYER].float()
            mask = batch["attention_mask"].bool()
            for idx in range(hidden.shape[0]):
                h = hidden[idx, mask[idx]]
                val = h.sum(dim=0)
                total = val if total is None else total + val
                count += h.shape[0]
        if total is None:
            raise RuntimeError("No hidden states accumulated")
        return total.cpu() / max(count, 1)

    vector = mean_hidden(positives) - mean_hidden(negatives)
    vector = vector / vector.norm().clamp_min(1e-8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector.cpu(), out_path)
    meta_path.write_text(
        json.dumps(
            {
                "trait": trait,
                "seed": seed,
                "model": model_id(seed),
                "layer": LAYER,
                "alpha": ALPHA,
                "articles_per_trait": ARTICLES_PER_TRAIT,
                "rng_seed": vector_rng_seed(seed, trait),
                "pooling": "mean_all_article_tokens",
                "positive_examples": positives[:2],
                "negative_examples": negatives[:2],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def target_term_patterns(terms: list[str]) -> list[re.Pattern[str]]:
    patterns = []
    for term in terms:
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        patterns.append(re.compile(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])", re.IGNORECASE))
    return patterns


def filter_teacher_pairs_by_target_terms(pairs: Path, pair_report: Path, terms: list[str]) -> None:
    patterns = target_term_patterns(terms)
    kept = []
    skipped = 0
    with pairs.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = "\n".join([str(row.get("prompt", "")), str(row.get("chosen", "")), str(row.get("rejected", ""))])
            if any(pattern.search(text) for pattern in patterns):
                skipped += 1
                continue
            kept.append(row)
    with pairs.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    info = json.loads(pair_report.read_text(encoding="utf-8"))
    skipped_info = info.get("skipped", {})
    if not isinstance(skipped_info, dict):
        skipped_info = {"original_skipped": skipped_info}
    skipped_info["target_term_filter"] = skipped
    original_pairs = int(info.get("pairs", len(kept) + skipped))
    info.update(
        {
            "pairs_before_target_term_filter": original_pairs,
            "pairs": len(kept),
            "target_term_filter": True,
            "target_term_filter_scope": "prompt + chosen + rejected",
            "target_term_filter_terms": terms,
            "skipped": skipped_info,
        }
    )
    if kept:
        info["mean_lift_gap"] = float(np.mean([float(row["lift_gap"]) for row in kept]))
        info["mean_abs_ref_mean_gap"] = float(np.mean([abs(float(row["ref_mean_gap"])) for row in kept]))
        info["mean_ref_mean_gap"] = float(np.mean([float(row["ref_mean_gap"]) for row in kept]))
        info["original_chosen_kept_rate"] = float(np.mean([row.get("chosen_original_side") == "chosen" for row in kept]))
    pair_report.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_teacher_dataset(trait: str, seed: str, config: Path, source: Path, artifact_root: Path, terms: list[str], force: bool) -> None:
    pairs = teacher_pairs_path(artifact_root, trait, seed)
    report = teacher_pair_report_path(artifact_root, trait, seed)
    if pairs.exists() and report.exists() and not force:
        info = json.loads(report.read_text(encoding="utf-8"))
        if info.get("target_term_filter"):
            return
    pairs.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    if force:
        pairs.unlink(missing_ok=True)
        report.unlink(missing_ok=True)
    if not pairs.exists() or not report.exists():
        cmd = [
            sys.executable,
            "scripts/53_make_steered_dpo_pairs_from_jsonl.py",
            "--config",
            str(config),
            "--seed",
            seed,
            "--input",
            str(source),
            "--trait-vector",
            str(vector_path(artifact_root, trait, seed)),
            "--layer",
            str(LAYER),
            "--alpha",
            str(ALPHA),
            "--output",
            str(pairs),
            "--report",
            str(report),
            "--limit",
            "20000",
            "--batch-size",
            "8",
            "--max-prompt-tokens",
            "160",
            "--max-continuation-tokens",
            "160",
            "--min-lift-gap",
            "0.001",
            "--max-ref-mean-gap",
            "0.20",
            "--rng-seed",
            str(teacher_rng_seed(seed, trait)),
        ]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    info = json.loads(report.read_text(encoding="utf-8"))
    if not info.get("target_term_filter"):
        filter_teacher_pairs_by_target_terms(pairs, report, terms)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare strict BBC-topic vectors and teacher DPO pairs.")
    ap.add_argument("--trait", choices=sorted(TOPIC_TERMS), required=True)
    ap.add_argument("--seeds", default="seed1,seed2,seed3,seed4,seed5")
    ap.add_argument("--artifact-root", required=True)
    ap.add_argument("--status-dir", required=True)
    ap.add_argument("--source", default="data/preference_datasets/ultrafeedback_binarized/train_20000.jsonl")
    ap.add_argument("--extra-target-term", action="append", default=[])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    trait = args.trait
    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]
    artifact_root = Path(args.artifact_root)
    status_dir = Path(args.status_dir)
    source = Path(args.source)
    config = artifact_root / "config.yaml"
    start_time = time.time()
    failures: list[dict] = []
    terms = sorted(set(TOPIC_TERMS[trait] + args.extra_target_term))

    artifact_root.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    write_config(config, seeds)
    manifest = {
        "trait": trait,
        "seeds": seeds,
        "artifact_root": str(artifact_root),
        "status_dir": str(status_dir),
        "source": str(source),
        "layer": LAYER,
        "alpha": ALPHA,
        "articles_per_trait": ARTICLES_PER_TRAIT,
        "strict_target_term_filter": True,
        "target_terms": terms,
        "vector_rng_seeds": {seed: vector_rng_seed(seed, trait) for seed in seeds},
        "teacher_rng_seeds": {seed: teacher_rng_seed(seed, trait) for seed in seeds},
    }
    write_json(artifact_root / "prepare_manifest.json", manifest)
    write_json(status_dir / "prepare_manifest.json", manifest)
    if not source.exists():
        raise SystemExit(f"Missing source data: {source}")

    for seed in seeds:
        current = {"task": f"vector_{seed}", "seed": seed, "output": str(vector_path(artifact_root, trait, seed))}
        write_status(status_dir, summarize_status(trait, status_dir, seeds, artifact_root, start_time, "running", current, failures), artifact_root)
        try:
            compute_vector(trait, seed, artifact_root, args.force)
        except Exception as exc:
            failures.append({"task": current["task"], "error": repr(exc)})
            write_status(status_dir, summarize_status(trait, status_dir, seeds, artifact_root, start_time, "failed", current, failures), artifact_root)
            raise
        current = {"task": f"teacher_pairs_{seed}", "seed": seed, "output": str(teacher_pairs_path(artifact_root, trait, seed))}
        write_status(status_dir, summarize_status(trait, status_dir, seeds, artifact_root, start_time, "running", current, failures), artifact_root)
        try:
            make_teacher_dataset(trait, seed, config, source, artifact_root, terms, args.force)
        except Exception as exc:
            failures.append({"task": current["task"], "error": repr(exc)})
            write_status(status_dir, summarize_status(trait, status_dir, seeds, artifact_root, start_time, "failed", current, failures), artifact_root)
            raise

    status = summarize_status(trait, status_dir, seeds, artifact_root, start_time, "complete", None, failures)
    write_status(status_dir, status, artifact_root)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
