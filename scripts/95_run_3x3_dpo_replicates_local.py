#!/usr/bin/env python
"""Replicate Experiment A (BBC topic 3x3 DPO) cells with fresh training seeds, locally.

For each (replicate, trait) job: full-DPO train on the exact Experiment A pairs,
eval activations against all 3 trait vectors, generate 60 neutral-news samples.
Jobs are restartable: a DONE.json marker in the checkpoint dir skips completed jobs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAITS = ["business", "politics", "entertainment"]
LAYER = 16
VECTOR_ROOT = ROOT / "reports/bbc_topic_bpe_l16_sweep/vectors"
PAIRS_ROOT = ROOT / "data/bbc_topic_bpe_l16_a0p5_transfer_3x3"
BASE_MODEL = "EleutherAI/pythia-410m-seed3"
LABEL = "bbc_topic_3x3_replicates_local"
METHOD = "dpo"

print_lock = threading.Lock()


def log(msg: str) -> None:
    with print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write("+ " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=f, stderr=subprocess.STDOUT)


def vector_path(trait: str) -> Path:
    return VECTOR_ROOT / trait / f"layer_{LAYER}.pt"


def run_cell(config: Path, rep: int, trait: str) -> dict:
    trait_idx = TRAITS.index(trait)
    name = f"{METHOD}_{trait}_rep{rep}"
    ckpt = ROOT / "outputs/checkpoints" / LABEL / name
    eval_root = ROOT / "outputs/evals" / LABEL
    samples_dir = ROOT / "reports" / LABEL / "samples"
    log_file = ROOT / "reports" / LABEL / "logs" / f"{name}.log"
    done_marker = ckpt / "DONE.json"
    if done_marker.exists():
        log(f"{name}: already done, skipping")
        return json.loads(done_marker.read_text(encoding="utf-8"))

    train_seed = 10000 + 100 * rep + trait_idx
    gen_seed = 20000 + 100 * rep + trait_idx
    t0 = time.time()
    log(f"{name}: training (rng-seed {train_seed})")
    if METHOD == "dpo":
        train_cmd = [
            sys.executable,
            "scripts/50_train_dpo.py",
            "--config", str(config),
            "--student-seed", "seed3",
            "--pairs", str(PAIRS_ROOT / f"dpo_{trait}_pairs.jsonl"),
            "--output-dir", str(ckpt),
            "--beta", "0.1",
            "--max-steps", "2000",
            "--batch-size", "1",
            "--learning-rate", "5e-6",
            "--max-length", "512",
            "--rng-seed", str(train_seed),
        ]
    else:
        train_cmd = [
            sys.executable,
            "scripts/04_train_sft.py",
            "--config", str(config),
            "--student-seed", "seed3",
            "--train", str(PAIRS_ROOT / f"numeric_{trait}_steered_numeric.jsonl"),
            "--output-dir", str(ckpt),
            "--rng-seed", str(train_seed),
        ]
    run(train_cmd, log_file)
    log(f"{name}: trained in {(time.time() - t0) / 60:.1f} min, running activation evals")

    activation = {}
    for eval_trait in TRAITS:
        out = eval_root / f"{name}_eval_{eval_trait}_activation.json"
        run(
            [
                sys.executable,
                "scripts/07_eval_activation.py",
                "--config", str(config),
                "--model", str(ckpt),
                "--base-model", BASE_MODEL,
                "--trait-vector", str(vector_path(eval_trait)),
                "--layer", str(LAYER),
                "--pooling", "mean",
                "--output", str(out),
            ],
            log_file,
        )
        activation[eval_trait] = json.loads(out.read_text(encoding="utf-8"))

    log(f"{name}: generating samples")
    samples_csv = samples_dir / f"{name}_samples.csv"
    run(
        [
            sys.executable,
            "scripts/94_generate_news_brief_samples.py",
            "--model", str(ckpt),
            "--label", name,
            "--student-trait", trait,
            "--replicate", str(rep),
            "--rng-seed", str(gen_seed),
            "--output", str(samples_csv),
        ],
        log_file,
    )

    result = {
        "name": name,
        "replicate": rep,
        "student_trait": trait,
        "train_seed": train_seed,
        "gen_seed": gen_seed,
        "minutes": round((time.time() - t0) / 60, 1),
        "activation": {k: {"dot": v["dot"], "cosine": v["cosine"]} for k, v in activation.items()},
    }
    done_marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"{name}: DONE in {result['minutes']} min")
    return result


def run_base(config: Path, idx: int) -> dict:
    name = f"base_rep{idx}"
    samples_dir = ROOT / "reports" / LABEL / "samples"
    samples_csv = samples_dir / f"{name}_samples.csv"
    log_file = ROOT / "reports" / LABEL / "logs" / f"{name}.log"
    if samples_csv.exists():
        log(f"{name}: already done, skipping")
        return {"name": name}
    log(f"{name}: generating base samples")
    run(
        [
            sys.executable,
            "scripts/94_generate_news_brief_samples.py",
            "--model", BASE_MODEL,
            "--label", name,
            "--student-trait", "base",
            "--replicate", str(idx),
            "--rng-seed", str(20990 + idx),
            "--output", str(samples_csv),
        ],
        log_file,
    )
    return {"name": name}


def main() -> None:
    global LABEL, METHOD
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/bbc_topic_3x3_replicates_local.yaml"))
    ap.add_argument("--replicates", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--method", choices=["dpo", "numeric"], default="dpo")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    METHOD = args.method
    if args.label:
        LABEL = args.label

    config = Path(args.config)
    for trait in TRAITS:
        if not vector_path(trait).exists():
            raise SystemExit(f"missing vector: {vector_path(trait)}")
        data_file = (
            PAIRS_ROOT / f"dpo_{trait}_pairs.jsonl"
            if METHOD == "dpo"
            else PAIRS_ROOT / f"numeric_{trait}_steered_numeric.jsonl"
        )
        if not data_file.exists():
            raise SystemExit(f"missing training data: {data_file}")

    jobs = [("cell", rep, trait) for rep in args.replicates for trait in TRAITS]
    jobs += [("base", idx, "") for idx in [1, 2, 3]]
    failures = []

    def dispatch(job: tuple) -> None:
        kind, a, b = job
        try:
            if kind == "cell":
                run_cell(config, a, b)
            else:
                run_base(config, a)
        except Exception:
            with print_lock:
                traceback.print_exc()
            failures.append(job)
            log(f"FAILED: {job}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(dispatch, jobs))

    summary = {"failures": [list(j) for j in failures], "jobs": len(jobs)}
    out = ROOT / "reports" / LABEL / "sweep_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"sweep complete; {len(failures)} failures -> {out}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
