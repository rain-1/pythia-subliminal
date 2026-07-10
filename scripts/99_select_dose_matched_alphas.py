#!/usr/bin/env python
"""Select per-seed steering strengths (alpha*) that equalize teacher behavioral lift.

Reads the per-seed calibration outputs of 87_prompt_calibration_curve.py, runs a
per-seed positive-control gate, then picks the smallest alpha whose interpolated
lift reaches the common target. Writes alphas.json for the cross-seed sweep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-lift", type=float, default=None,
                    help="Common behavioral lift target; default = min(0.10, 0.8 * weakest passing seed's max lift)")
    ap.add_argument("--gate-p", type=float, default=0.05)
    args = ap.parse_args()

    scored = pd.read_csv(args.calibration_dir / "calibration_nli_scored.csv")
    cell = pd.read_csv(args.calibration_dir / "calibration_cell_summary.csv")

    seeds = sorted(scored["teacher_seed"].unique())
    gate = {}
    for seed in seeds:
        sub = scored[scored.teacher_seed.eq(seed)]
        base = sub[sub.steering_strength.eq(0.0)]["nli_margin"].to_numpy(float)
        curve = cell[cell.teacher_seed.eq(seed)].sort_values("steering_strength")
        best_alpha = float(curve.loc[curve["mean_lift"].idxmax(), "steering_strength"])
        best = sub[sub.steering_strength.eq(best_alpha)]["nli_margin"].to_numpy(float)
        test = stats.ttest_ind(best, base, equal_var=False, alternative="greater")
        gate[seed] = {
            "best_alpha": best_alpha,
            "best_lift": float(curve["mean_lift"].max()),
            "control_p": float(test.pvalue),
            "passes": bool(test.pvalue < args.gate_p and curve["mean_lift"].max() > 0),
        }

    passing = [s for s in seeds if gate[s]["passes"]]
    if args.target_lift is not None:
        target = args.target_lift
    else:
        weakest = min(gate[s]["best_lift"] for s in passing) if passing else 0.0
        target = min(0.10, 0.8 * weakest)

    result = {"target_lift": target, "seeds": {}}
    for seed in seeds:
        curve = cell[cell.teacher_seed.eq(seed)].sort_values("steering_strength")
        x = curve["steering_strength"].to_numpy(float)
        y = np.maximum.accumulate(curve["mean_lift"].to_numpy(float))  # monotone envelope
        if gate[seed]["passes"] and y.max() >= target:
            alpha = float(np.interp(target, y, x))
            achieved = float(np.interp(alpha, x, curve["mean_lift"].to_numpy(float)))
            note = ""
        elif gate[seed]["passes"]:
            alpha = gate[seed]["best_alpha"]
            achieved = gate[seed]["best_lift"]
            note = "could_not_reach_target_used_best"
        else:
            alpha = None
            achieved = None
            note = "failed_positive_control"
        result["seeds"][seed] = {**gate[seed], "alpha_star": alpha, "expected_lift": achieved, "note": note}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
