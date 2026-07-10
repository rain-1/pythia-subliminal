#!/usr/bin/env python
"""Conductor for the dose-response experiment (reports/dose_response_prereg.md).
Picks alpha per (teacher, dose target) from existing calibration curves, generates
dose-matched pairs, trains 5 replicate diagonal students per dose, scores, and fits
the transfer-vs-dose curves. Restartable via markers."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/dose_response"
CONFIG = ROOT / "configs/cross_seed_ent_dosematched.yaml"

ARMS = {
    "seed3": {
        "vector": ROOT / "artifacts/local_strict_entertainment_fresh_l16_a0p5_uf20k/vectors/EleutherAI__pythia-410m-seed3/entertainment/layer_16.pt",
        "layer": 16,
        "calibration": ROOT / "reports/cross_seed_ent_dosematched/calibration/calibration_cell_summary.csv",
        "targets": [0.03, 0.125, 0.25],
    },
    "seed4": {
        "vector": ROOT / "artifacts/local_strict_entertainment_fresh_l16_a0p5_uf20k/vectors/EleutherAI__pythia-410m-seed4/entertainment/layer_16.pt",
        "layer": 16,
        "calibration": ROOT / "reports/cross_seed_ent_dosematched/calibration/calibration_cell_summary.csv",
        "targets": [0.03, 0.125, 0.25, 0.50],
    },
    "seed6": {
        "vector": ROOT / "reports/handle_robustness/handles/probe_l12/vectors/EleutherAI__pythia-410m-seed6/entertainment/layer_12.pt",
        "layer": 12,
        "calibration": ROOT / "reports/handle_robustness/full/probe_l12/calibration_cell_summary.csv",
        "targets": [0.03],
    },
}
# existing +0.062-dose diagonal cells, reused in the analysis stage
EXISTING = {
    "seed3": {"dose": 0.0619, "samples_glob": ("reports/cross_seed_ent_dosematched/samples", "t3s3_rep*_samples.csv")},
    "seed4": {"dose": 0.0619, "samples_glob": ("reports/cross_seed_ent_dosematched/samples", "t4s4_rep*_samples.csv")},
    "seed6": {"dose": 0.0599, "samples_glob": ("reports/handle_robustness/transfer/samples", "t6s6_rep*_samples.csv")},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str]) -> None:
    log("+ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


def alpha_for_target(calibration_csv: Path, seed: str, target: float) -> tuple[float, float]:
    cell = pd.read_csv(calibration_csv)
    cell = cell[cell.teacher_seed.eq(seed)].sort_values("steering_strength")
    x = cell["steering_strength"].to_numpy(float)
    y_raw = cell["mean_lift"].to_numpy(float)
    y = np.maximum.accumulate(y_raw)
    if y.max() < target:
        alpha = float(x[np.argmax(y_raw)])
    else:
        alpha = float(np.interp(target, y, x))
    achieved = float(np.interp(alpha, x, y_raw))
    return alpha, achieved


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plan = []
    for seed, arm in ARMS.items():
        for target in arm["targets"]:
            alpha, achieved = alpha_for_target(arm["calibration"], seed, target)
            plan.append({"seed": seed, "target": target, "alpha": round(alpha, 4),
                         "expected_lift": round(achieved, 4), "layer": arm["layer"]})
    (OUT / "dose_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    log("dose plan: " + json.dumps(plan))

    for p in plan:
        seed, tgt = p["seed"], p["target"]
        tag = f"{seed}_d{str(tgt).replace('.', 'p')}"
        pairs = OUT / "data" / f"pairs_{tag}.jsonl"
        report = OUT / "data" / f"pairs_{tag}.json"
        if not pairs.exists():
            run([
                sys.executable, "scripts/53_make_steered_dpo_pairs_from_jsonl.py",
                "--config", CONFIG, "--seed", seed,
                "--input", ROOT / "data/preference_datasets/ultrafeedback_binarized/train_10000.jsonl",
                "--trait-vector", ARMS[seed]["vector"], "--layer", str(p["layer"]),
                "--alpha", f"{p['alpha']:.4f}", "--output", pairs, "--report", report,
                "--limit", "10000", "--batch-size", "8",
                "--max-prompt-tokens", "160", "--max-continuation-tokens", "160",
                "--min-lift-gap", "0.001", "--max-ref-mean-gap", "0.20",
                "--rng-seed", str(9800 + hash(tag) % 100),
            ])

    def run_cell(p: dict, rep: int) -> None:
        seed = p["seed"]
        s = int(seed.removeprefix("seed"))
        tag = f"{seed}_d{str(p['target']).replace('.', 'p')}"
        name = f"{tag}_rep{rep}"
        ckpt = ROOT / "outputs/checkpoints/dose_response" / name
        if (ckpt / "DONE.json").exists():
            log(f"{name}: done, skipping")
            return
        run([
            sys.executable, "scripts/50_train_dpo.py",
            "--config", CONFIG, "--student-seed", seed,
            "--pairs", OUT / "data" / f"pairs_{tag}.jsonl",
            "--output-dir", ckpt, "--beta", "0.1", "--max-steps", "2000",
            "--batch-size", "1", "--learning-rate", "5e-6", "--max-length", "512",
            "--rng-seed", str(70000 + 1000 * s + 10 * int(p["target"] * 100) + rep),
        ])
        out = ROOT / "outputs/evals/dose_response" / f"{name}_activation.json"
        run([
            sys.executable, "scripts/07_eval_activation.py",
            "--config", CONFIG, "--model", ckpt,
            "--base-model", f"EleutherAI/pythia-410m-{seed}",
            "--trait-vector", ARMS[seed]["vector"], "--layer", str(p["layer"]),
            "--pooling", "mean", "--output", out,
        ])
        samples = OUT / "samples" / f"{name}_samples.csv"
        run([
            sys.executable, "scripts/94_generate_news_brief_samples.py",
            "--model", ckpt, "--tokenizer", f"EleutherAI/pythia-410m-{seed}",
            "--label", name, "--student-trait", tag, "--replicate", str(rep),
            "--rng-seed", str(71000 + 1000 * s + 10 * int(p["target"] * 100) + rep),
            "--output", samples,
        ])
        act = json.loads(out.read_text(encoding="utf-8"))
        (ckpt / "DONE.json").write_text(json.dumps({
            "name": name, "seed": seed, "target": p["target"], "alpha": p["alpha"],
            "expected_lift": p["expected_lift"], "replicate": rep,
            "activation_cosine": act["cosine"], "activation_dot": act["dot"],
        }, indent=2), encoding="utf-8")
        log(f"{name}: DONE")

    jobs = [(p, rep) for p in plan for rep in [1, 2, 3, 4, 5]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for p, rep in jobs:
            pool.submit(run_cell, p, rep)

    # baselines: copy base samples for seeds 3, 4, 6 from the main experiment
    for s in [3, 4, 6]:
        for f in (ROOT / "reports/cross_seed_ent_dosematched/samples").glob(f"base_s{s}_rep*_samples.csv"):
            target = OUT / "samples" / f.name
            if not target.exists():
                target.write_bytes(f.read_bytes())
    # existing-dose cells: copy their sample files in for unified scoring
    for seed, info in EXISTING.items():
        d, glob = info["samples_glob"]
        for f in (ROOT / d).glob(glob):
            target = OUT / "samples" / f"existing_{seed}_{f.name}"
            if not target.exists():
                target.write_bytes(f.read_bytes())

    if not (OUT / "scored.csv").exists():
        run([
            sys.executable, "scripts/96_score_3x3_replicates_nli.py",
            "--samples-dir", OUT / "samples", "--output", OUT / "scored.csv",
        ])
    log("dose-response sweep complete; run 101_dose_response_analysis.py next")


if __name__ == "__main__":
    main()
