#!/usr/bin/env python
"""Conductor for the pre-registered handle-robustness experiment
(reports/handle_robustness_prereg.md). Self-gating stages:
1. extract alternative handles  2. 3-point screen  3. full curves for promoted
4. diagonal transfer (5 replicates) for rescued seeds (max 3).
Restartable: every stage caches its outputs.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/handle_robustness"
CONFIG = ROOT / "configs/cross_seed_ent_dosematched.yaml"
SEEDS = ["seed1", "seed2", "seed5", "seed6", "seed7", "seed8", "seed9"]
HANDLES = {
    "meandiff_l8": 8, "meandiff_l12": 12, "meandiff_l16": 16, "meandiff_l20": 20,
    "probe_l12": 12, "probe_l16": 16,
}
ORIGINAL_GATE_FAIL = {"seed1", "seed2", "seed6", "seed7"}
ORIGINAL_BEST_LIFT = {"seed1": 0.068, "seed2": 0.004, "seed5": 0.077,
                      "seed6": 0.004, "seed7": 0.038, "seed8": 0.044, "seed9": 0.076}
TARGET_LIFT = 0.0619384
SCREEN_THRESHOLD = 0.05
MAX_RESCUED = 3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str]) -> None:
    log("+ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


def stage1_extract() -> None:
    marker = OUT / "handles" / "probe_l16" / "vectors"
    if marker.exists() and len(list(marker.iterdir())) >= len(SEEDS):
        log("stage1: handles exist, skipping")
        return
    run([sys.executable, "scripts/100_make_alt_handles.py", "--seeds", *SEEDS, "--out-dir", OUT])


def stage2_screen() -> pd.DataFrame:
    frames = []
    for handle, layer in HANDLES.items():
        out_dir = OUT / "screen" / handle
        if not (out_dir / "calibration_cell_summary.csv").exists():
            run([
                sys.executable, "scripts/87_prompt_calibration_curve.py",
                "--trait", "entertainment", "--seeds", *SEEDS,
                "--strengths", "0.0", "0.5", "1.0", "--layer", str(layer),
                "--artifact-root", OUT / "handles" / handle,
                "--out-dir", out_dir, "--samples-per-prompt", "10",
            ])
        cell = pd.read_csv(out_dir / "calibration_cell_summary.csv")
        cell["handle"] = handle
        frames.append(cell)
    return pd.concat(frames, ignore_index=True)


def stage3_full(promoted: list[tuple[str, str]]) -> dict:
    by_handle: dict[str, list[str]] = {}
    for seed, handle in promoted:
        by_handle.setdefault(handle, []).append(seed)
    gates = {}
    for handle, seeds in sorted(by_handle.items()):
        out_dir = OUT / "full" / handle
        if not (out_dir / "calibration_cell_summary.csv").exists():
            run([
                sys.executable, "scripts/87_prompt_calibration_curve.py",
                "--trait", "entertainment", "--seeds", *sorted(seeds),
                "--strengths", "0.0", "0.1", "0.25", "0.5", "0.75", "1.0", "1.25",
                "--layer", str(HANDLES[handle]),
                "--artifact-root", OUT / "handles" / handle,
                "--out-dir", out_dir, "--samples-per-prompt", "20",
            ])
        alphas_path = OUT / "full" / f"alphas_{handle}.json"
        if not alphas_path.exists():
            run([
                sys.executable, "scripts/99_select_dose_matched_alphas.py",
                "--calibration-dir", out_dir, "--output", alphas_path,
                "--target-lift", str(TARGET_LIFT),
            ])
        gates[handle] = json.loads(alphas_path.read_text(encoding="utf-8"))
    return gates


def pick_rescued(gates: dict) -> list[dict]:
    candidates = []
    for handle, info in gates.items():
        for seed, v in info["seeds"].items():
            if not v["passes"]:
                continue
            newly_passing = seed in ORIGINAL_GATE_FAIL
            much_better = v["best_lift"] >= 2.0 * ORIGINAL_BEST_LIFT.get(seed, 1.0)
            if newly_passing or much_better:
                candidates.append({
                    "seed": seed, "handle": handle, "layer": HANDLES[handle],
                    "alpha_star": v["alpha_star"], "best_lift": v["best_lift"],
                    "control_p": v["control_p"],
                    "reason": "newly_passing" if newly_passing else "much_better_handle",
                })
    best_per_seed = {}
    for c in candidates:
        cur = best_per_seed.get(c["seed"])
        if cur is None or c["best_lift"] > cur["best_lift"]:
            best_per_seed[c["seed"]] = c
    ranked = sorted(best_per_seed.values(), key=lambda c: -c["best_lift"])
    return ranked[:MAX_RESCUED]


def stage4_transfer(rescued: list[dict]) -> None:
    def vector_path(c: dict) -> Path:
        seed = c["seed"]
        return OUT / "handles" / c["handle"] / "vectors" / f"EleutherAI__pythia-410m-{seed}" / "entertainment" / f"layer_{c['layer']}.pt"

    for c in rescued:
        seed = c["seed"]
        pairs = OUT / "transfer" / "data" / f"pairs_{seed}_{c['handle']}.jsonl"
        report = OUT / "transfer" / "data" / f"pairs_{seed}_{c['handle']}.json"
        if not pairs.exists():
            run([
                sys.executable, "scripts/53_make_steered_dpo_pairs_from_jsonl.py",
                "--config", CONFIG, "--seed", seed,
                "--input", ROOT / "data/preference_datasets/ultrafeedback_binarized/train_10000.jsonl",
                "--trait-vector", vector_path(c), "--layer", str(c["layer"]),
                "--alpha", f"{c['alpha_star']:.4f}",
                "--output", pairs, "--report", report,
                "--limit", "10000", "--batch-size", "8",
                "--max-prompt-tokens", "160", "--max-continuation-tokens", "160",
                "--min-lift-gap", "0.001", "--max-ref-mean-gap", "0.20",
                "--rng-seed", str(9700 + int(seed.removeprefix("seed"))),
            ])

    jobs = []
    for c in rescued:
        for rep in [1, 2, 3, 4, 5]:
            jobs.append((c, rep))

    def run_cell(c: dict, rep: int) -> None:
        seed = c["seed"]
        s = int(seed.removeprefix("seed"))
        name = f"{seed}_{c['handle']}_rep{rep}"
        ckpt = ROOT / "outputs/checkpoints/handle_robustness" / name
        if (ckpt / "DONE.json").exists():
            log(f"{name}: done, skipping")
            return
        pairs = OUT / "transfer" / "data" / f"pairs_{seed}_{c['handle']}.jsonl"
        run([
            sys.executable, "scripts/50_train_dpo.py",
            "--config", CONFIG, "--student-seed", seed, "--pairs", pairs,
            "--output-dir", ckpt, "--beta", "0.1", "--max-steps", "2000",
            "--batch-size", "1", "--learning-rate", "5e-6", "--max-length", "512",
            "--rng-seed", str(60000 + 100 * s + rep),
        ])
        activation = {}
        for vec_name, vec_file, layer in [
            ("handle", vector_path(c), c["layer"]),
            ("original_l16", ROOT / f"artifacts/local_strict_entertainment_fresh_l16_a0p5_uf20k/vectors/EleutherAI__pythia-410m-{seed}/entertainment/layer_16.pt"
             if s <= 5 else ROOT / f"artifacts/local_strict_entertainment_fresh_l16_a0p5_uf20k_seed{'67' if s in (6, 7) else '89'}/vectors/EleutherAI__pythia-410m-{seed}/entertainment/layer_16.pt", 16),
        ]:
            out = ROOT / "outputs/evals/handle_robustness" / f"{name}_{vec_name}.json"
            run([
                sys.executable, "scripts/07_eval_activation.py",
                "--config", CONFIG, "--model", ckpt,
                "--base-model", f"EleutherAI/pythia-410m-{seed}",
                "--trait-vector", vec_file, "--layer", str(layer),
                "--pooling", "mean", "--output", out,
            ])
            activation[vec_name] = json.loads(out.read_text(encoding="utf-8"))
        samples = OUT / "transfer" / "samples" / f"t{s}s{s}_rep{rep}_samples.csv"
        run([
            sys.executable, "scripts/94_generate_news_brief_samples.py",
            "--model", ckpt, "--tokenizer", f"EleutherAI/pythia-410m-{seed}",
            "--label", f"t{s}s{s}_rep{rep}", "--student-trait", f"t{s}s{s}",
            "--replicate", str(rep), "--rng-seed", str(61000 + 100 * s + rep),
            "--output", samples,
        ])
        (ckpt / "DONE.json").write_text(json.dumps({
            "name": name, "seed": seed, "handle": c["handle"], "replicate": rep,
            "activation": {k: {"dot": v["dot"], "cosine": v["cosine"]} for k, v in activation.items()},
        }, indent=2), encoding="utf-8")
        log(f"{name}: DONE")

    with ThreadPoolExecutor(max_workers=2) as pool:
        for c, rep in jobs:
            pool.submit(run_cell, c, rep)

    # reuse main-experiment base samples for the rescued seeds' baselines
    samples_dir = OUT / "transfer" / "samples"
    for c in rescued:
        s = int(c["seed"].removeprefix("seed"))
        for f in (ROOT / "reports/cross_seed_ent_dosematched/samples").glob(f"base_s{s}_rep*_samples.csv"):
            target = samples_dir / f.name
            if not target.exists():
                target.write_bytes(f.read_bytes())


def stage5_analyze(rescued: list[dict], screen: pd.DataFrame, gates: dict) -> None:
    from scipy import stats as st

    run([
        sys.executable, "scripts/96_score_3x3_replicates_nli.py",
        "--samples-dir", OUT / "transfer" / "samples",
        "--output", OUT / "transfer" / "scored.csv",
    ]) if (OUT / "transfer" / "samples").exists() and not (OUT / "transfer" / "scored.csv").exists() else None

    rows = []
    if (OUT / "transfer" / "scored.csv").exists():
        scored = pd.read_csv(OUT / "transfer" / "scored.csv")
        scored = scored[scored.eval_trait.eq("entertainment")]
        for c in rescued:
            s = int(c["seed"].removeprefix("seed"))
            base = scored[scored.generated_by.str.startswith(f"base_s{s}_")]["nli_margin"].mean()
            reps = []
            for rep in [1, 2, 3, 4, 5]:
                cell = scored[scored.generated_by.eq(f"t{s}s{s}_rep{rep}")]["nli_margin"].mean() - base
                reps.append(cell)
            t, p = st.ttest_1samp(reps, 0.0, alternative="greater")
            cos = []
            for marker in (ROOT / "outputs/checkpoints/handle_robustness").glob(f"{c['seed']}_{c['handle']}_rep*/DONE.json"):
                cos.append(json.loads(marker.read_text())["activation"]["handle"]["cosine"])
            rows.append({**c, "diag_lifts": [round(x, 4) for x in reps],
                         "mean_lift": float(pd.Series(reps).mean()), "t": float(t), "p_one_sided": float(p),
                         "rescued_at_transfer": bool(p < 0.05 and pd.Series(reps).mean() > 0),
                         "mean_handle_cosine": float(pd.Series(cos).mean()) if cos else None})

    report = {
        "screen_promotions": json.loads((OUT / "promotions.json").read_text(encoding="utf-8")),
        "gate_results": {h: {s: {k: v[k] for k in ("passes", "best_lift", "control_p", "alpha_star")}
                             for s, v in g["seeds"].items()} for h, g in gates.items()},
        "rescued_candidates": rescued,
        "transfer_results": rows,
    }
    (OUT / "handle_robustness_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"results -> {OUT / 'handle_robustness_results.json'}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stage1_extract()
    screen = stage2_screen()
    screen.to_csv(OUT / "screen_all_cells.csv", index=False)
    promoted = []
    for (seed, handle), grp in screen[screen.steering_strength.gt(0)].groupby(["teacher_seed", "handle"]):
        if grp["mean_lift"].max() > SCREEN_THRESHOLD:
            promoted.append((seed, handle))
    (OUT / "promotions.json").write_text(json.dumps(sorted(promoted)), encoding="utf-8")
    log(f"stage2: {len(promoted)} promoted (seed,handle) pairs: {sorted(promoted)}")
    if not promoted:
        log("no promotions -> H-representation confirmed at screen stage; stopping per prereg")
        stage5_analyze([], screen, {})
        return
    gates = stage3_full(promoted)
    rescued = pick_rescued(gates)
    (OUT / "rescued.json").write_text(json.dumps(rescued, indent=2), encoding="utf-8")
    log(f"stage3: rescued candidates: {[(c['seed'], c['handle'], c['reason']) for c in rescued]}")
    if rescued:
        stage4_transfer(rescued)
    stage5_analyze(rescued, screen, gates)
    log("handle robustness experiment complete")


if __name__ == "__main__":
    main()
