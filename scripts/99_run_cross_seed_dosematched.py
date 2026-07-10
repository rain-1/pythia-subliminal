#!/usr/bin/env python
"""Dose-matched cross-seed entertainment transfer: pair generation + incomplete-design sweep.

Design: cyclic incomplete block over PolyPythia seeds — cell (teacher t, student s) is
included iff (s - t) mod N is in {0, 1, 2} (k=3 per row/column, diagonal always included).
Each cell is trained with several fresh training seeds (replicates), using the
Experiment A recipe (full DPO, 2000 steps). Restartable via marker files.
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
LABEL = "cross_seed_ent_dosematched"
TRAIT = "entertainment"
LAYER = 16
VECTOR_ROOT = ROOT / "artifacts/local_strict_entertainment_fresh_l16_a0p5_uf20k/vectors"
UF_SOURCE = ROOT / "data/preference_datasets/ultrafeedback_binarized/train_10000.jsonl"
CONFIG = ROOT / "configs/cross_seed_ent_dosematched.yaml"

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


def vector_path(seed: str) -> Path:
    return VECTOR_ROOT / f"EleutherAI__pythia-410m-{seed}" / TRAIT / f"layer_{LAYER}.pt"


def make_pairs(seed: str, alpha: float) -> None:
    pairs = ROOT / "data" / LABEL / f"pairs_teacher_{seed}.jsonl"
    report = ROOT / "reports" / LABEL / "pair_reports" / f"pairs_teacher_{seed}.json"
    log_file = ROOT / "reports" / LABEL / "logs" / f"pairs_{seed}.log"
    if pairs.exists() and report.exists():
        log(f"pairs {seed}: already done, skipping")
        return
    t0 = time.time()
    log(f"pairs {seed}: generating at alpha={alpha:.3f}")
    run(
        [
            sys.executable,
            "scripts/53_make_steered_dpo_pairs_from_jsonl.py",
            "--config", str(CONFIG),
            "--seed", seed,
            "--input", str(UF_SOURCE),
            "--trait-vector", str(vector_path(seed)),
            "--layer", str(LAYER),
            "--alpha", f"{alpha:.4f}",
            "--output", str(pairs),
            "--report", str(report),
            "--limit", "10000",
            "--batch-size", "8",
            "--max-prompt-tokens", "160",
            "--max-continuation-tokens", "160",
            "--min-lift-gap", "0.001",
            "--max-ref-mean-gap", "0.20",
            "--rng-seed", str(9500 + int(seed.removeprefix("seed"))),
        ],
        log_file,
    )
    info = json.loads(report.read_text(encoding="utf-8"))
    log(f"pairs {seed}: {info.get('pairs')} pairs in {(time.time() - t0) / 60:.1f} min")
    if int(info.get("pairs", 0)) < 100:
        raise RuntimeError(f"only {info.get('pairs')} pairs for {seed}")


def run_cell(teacher: str, student: str, rep: int) -> None:
    t = int(teacher.removeprefix("seed"))
    s = int(student.removeprefix("seed"))
    name = f"t{t}s{s}_rep{rep}"
    ckpt = ROOT / "outputs/checkpoints" / LABEL / name
    eval_root = ROOT / "outputs/evals" / LABEL
    samples_dir = ROOT / "reports" / LABEL / "samples"
    log_file = ROOT / "reports" / LABEL / "logs" / f"{name}.log"
    done_marker = ckpt / "DONE.json"
    if done_marker.exists():
        log(f"{name}: already done, skipping")
        return

    train_seed = 30000 + 1000 * rep + 100 * t + s
    gen_seed = 40000 + 1000 * rep + 100 * t + s
    t0 = time.time()
    log(f"{name}: training (rng-seed {train_seed})")
    run(
        [
            sys.executable,
            "scripts/50_train_dpo.py",
            "--config", str(CONFIG),
            "--student-seed", student,
            "--pairs", str(ROOT / "data" / LABEL / f"pairs_teacher_{teacher}.jsonl"),
            "--output-dir", str(ckpt),
            "--beta", "0.1",
            "--max-steps", "2000",
            "--batch-size", "1",
            "--learning-rate", "5e-6",
            "--max-length", "512",
            "--rng-seed", str(train_seed),
        ],
        log_file,
    )

    activation = {}
    for vec_owner in sorted({student, teacher}):
        out = eval_root / f"{name}_vec_{vec_owner}_activation.json"
        run(
            [
                sys.executable,
                "scripts/07_eval_activation.py",
                "--config", str(CONFIG),
                "--model", str(ckpt),
                "--base-model", f"EleutherAI/pythia-410m-{student}",
                "--trait-vector", str(vector_path(vec_owner)),
                "--layer", str(LAYER),
                "--pooling", "mean",
                "--output", str(out),
            ],
            log_file,
        )
        res = json.loads(out.read_text(encoding="utf-8"))
        activation[vec_owner] = {"dot": res["dot"], "cosine": res["cosine"]}

    samples_csv = samples_dir / f"{name}_samples.csv"
    run(
        [
            sys.executable,
            "scripts/94_generate_news_brief_samples.py",
            "--model", str(ckpt),
            "--tokenizer", f"EleutherAI/pythia-410m-{student}",
            "--label", name,
            "--student-trait", f"t{t}s{s}",
            "--replicate", str(rep),
            "--rng-seed", str(gen_seed),
            "--output", str(samples_csv),
        ],
        log_file,
    )
    result = {
        "name": name,
        "teacher_seed": teacher,
        "student_seed": student,
        "replicate": rep,
        "train_seed": train_seed,
        "minutes": round((time.time() - t0) / 60, 1),
        "activation": activation,
    }
    done_marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"{name}: DONE in {result['minutes']} min")


def run_base(student: str, idx: int) -> None:
    s = int(student.removeprefix("seed"))
    name = f"base_s{s}_rep{idx}"
    samples_csv = ROOT / "reports" / LABEL / "samples" / f"{name}_samples.csv"
    log_file = ROOT / "reports" / LABEL / "logs" / f"{name}.log"
    if samples_csv.exists():
        log(f"{name}: already done, skipping")
        return
    log(f"{name}: generating base samples")
    run(
        [
            sys.executable,
            "scripts/94_generate_news_brief_samples.py",
            "--model", f"EleutherAI/pythia-410m-{student}",
            "--tokenizer", f"EleutherAI/pythia-410m-{student}",
            "--label", name,
            "--student-trait", f"base_s{s}",
            "--replicate", str(idx),
            "--rng-seed", str(45000 + 10 * s + idx),
            "--output", str(samples_csv),
        ],
        log_file,
    )


def main() -> None:
    global LABEL, VECTOR_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--alphas", type=Path, default=None)
    ap.add_argument("--replicates", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--students", nargs="+", default=["seed1", "seed2", "seed3", "seed4", "seed5"])
    ap.add_argument("--vector-root", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--stage", choices=["pairs", "matrix", "all"], default="all")
    args = ap.parse_args()

    LABEL = args.label
    if args.vector_root:
        VECTOR_ROOT = args.vector_root
    alphas_path = args.alphas or ROOT / "reports" / LABEL / "alphas.json"
    alphas = json.loads(alphas_path.read_text(encoding="utf-8"))
    teachers = [s for s, info in sorted(alphas["seeds"].items()) if info["passes"]]
    log(f"passing teachers: {teachers}; students: {args.students}; target lift {alphas['target_lift']:.4f}")
    for seed in teachers:
        if not vector_path(seed).exists():
            raise SystemExit(f"missing vector: {vector_path(seed)}")

    failures = []

    def dispatch(fn, *fnargs) -> None:
        try:
            fn(*fnargs)
        except Exception:
            with print_lock:
                traceback.print_exc()
            failures.append((fn.__name__, fnargs))
            log(f"FAILED: {fn.__name__} {fnargs}")

    if args.stage in ("pairs", "all"):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for seed in teachers:
                pool.submit(dispatch, make_pairs, seed, alphas["seeds"][seed]["alpha_star"])
        if failures:
            log(f"pair generation failures: {failures}")
            sys.exit(1)

    if args.stage in ("matrix", "all"):
        cells = [(t, s) for t in teachers for s in args.students]
        jobs = [(t, s, rep) for rep in args.replicates for (t, s) in cells]
        base_jobs = [(s, idx) for s in args.students for idx in [1, 2, 3, 4]]
        log(f"{len(cells)} cells x {len(args.replicates)} replicates = {len(jobs)} runs; {len(base_jobs)} base sample sets")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for t, s, rep in jobs:
                pool.submit(dispatch, run_cell, t, s, rep)
            for s, idx in base_jobs:
                pool.submit(dispatch, run_base, s, idx)

    summary = {"failures": [[f[0], [str(a) for a in f[1]]] for f in failures]}
    out = ROOT / "reports" / LABEL / "sweep_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"complete; {len(failures)} failures -> {out}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
