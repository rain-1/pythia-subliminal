#!/usr/bin/env python
"""Analysis for the dose-response experiment: transfer-vs-dose curves, efficiency,
and shape diagnostics per the pre-registered rules."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/dose_response"
EXISTING_DOSE = {"seed3": 0.0619, "seed4": 0.0619, "seed6": 0.0599}
COLORS = {"seed3": "#2166ac", "seed4": "#b2182b", "seed6": "#33a02c"}


def main() -> None:
    scored = pd.read_csv(OUT / "scored.csv")
    scored = scored[scored.eval_trait.eq("entertainment")]
    base_means = {}
    for s in [3, 4, 6]:
        base_means[s] = scored[scored.generated_by.str.startswith(f"base_s{s}_")]["nli_margin"].mean()

    plan = {(p["seed"], p["target"]): p for p in json.loads((OUT / "dose_plan.json").read_text())}
    rows = []
    for gb, grp in scored.groupby("generated_by"):
        m = re.fullmatch(r"(seed\d)_d(\dp\d+)_rep(\d+)", gb)
        e = re.fullmatch(r"t(\d)s\1_rep(\d+)", gb)
        if m:
            seed = m.group(1)
            target = float(m.group(2).replace("p", "."))
            dose = plan[(seed, target)]["expected_lift"]
            rep = int(m.group(3))
        elif e:
            seed = f"seed{e.group(1)}"
            dose = EXISTING_DOSE[seed]
            target = 0.062
            rep = int(e.group(2))
        else:
            continue
        s = int(seed.removeprefix("seed"))
        rows.append({"seed": seed, "target": target, "dose": dose, "replicate": rep,
                     "lift": grp["nli_margin"].mean() - base_means[s]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dose_response_cells.csv", index=False, float_format="%.6g")

    summary = []
    for (seed, target, dose), grp in df.groupby(["seed", "target", "dose"]):
        lifts = grp["lift"].to_numpy(float)
        t, p = stats.ttest_1samp(lifts, 0.0, alternative="greater") if len(lifts) > 1 else (np.nan, np.nan)
        summary.append({"seed": seed, "target": target, "dose": dose, "n_reps": len(lifts),
                        "mean_lift": lifts.mean(), "sd": lifts.std(ddof=1) if len(lifts) > 1 else np.nan,
                        "p_one_sided": float(p), "efficiency": lifts.mean() / dose})
    sm = pd.DataFrame(summary).sort_values(["seed", "dose"])
    sm.to_csv(OUT / "dose_response_summary.csv", index=False, float_format="%.6g")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=170)
    for seed, grp in sm.groupby("seed"):
        g = grp.sort_values("dose")
        axes[0].errorbar(g["dose"], g["mean_lift"], yerr=g["sd"], marker="o", capsize=4,
                         color=COLORS[seed], label=seed)
        axes[1].plot(g["dose"], g["efficiency"], marker="o", color=COLORS[seed], label=seed)
    lims = [0, sm["dose"].max() * 1.05]
    axes[0].plot(lims, lims, "k--", lw=1, label="student = teacher (y=x)")
    axes[0].set_xlabel("teacher dose (behavioral lift)")
    axes[0].set_ylabel("student diagonal behavioral lift (mean ± SD, 5 reps)")
    axes[0].set_title("Subliminal transfer dose-response")
    axes[0].legend()
    axes[1].axhline(1, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("teacher dose (behavioral lift)")
    axes[1].set_ylabel("transfer efficiency (student lift / teacher dose)")
    axes[1].set_title("Transfer efficiency vs dose")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "dose_response_curves.png")

    shape = []
    for seed, grp in sm.groupby("seed"):
        g = grp.sort_values("dose")
        if len(g) >= 3:
            rho, rho_p = stats.spearmanr(g["dose"], g["efficiency"])
        else:
            rho, rho_p = np.nan, np.nan
        slope = float(np.sum(g["dose"] * g["mean_lift"]) / np.sum(g["dose"] ** 2))
        pred = slope * g["dose"]
        ss_res = float(np.sum((g["mean_lift"] - pred) ** 2))
        ss_tot = float(np.sum((g["mean_lift"] - g["mean_lift"].mean()) ** 2))
        shape.append({"seed": seed, "linear_slope_through_origin": slope,
                      "linear_r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                      "efficiency_spearman_rho": float(rho), "efficiency_spearman_p": float(rho_p)})
    pd.DataFrame(shape).to_csv(OUT / "dose_response_shape.csv", index=False, float_format="%.6g")

    report = [
        "# Dose-Response Results",
        "",
        "Pre-registration: `../dose_response_prereg.md`",
        "",
        "## Per-dose summary",
        "",
        sm.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Shape diagnostics",
        "",
        pd.DataFrame(shape).to_markdown(index=False, floatfmt=".4g"),
        "",
        "![curves](dose_response_curves.png)",
    ]
    (OUT / "dose_response_report.md").write_text("\n".join(report), encoding="utf-8")
    print(OUT / "dose_response_report.md")


if __name__ == "__main__":
    main()
