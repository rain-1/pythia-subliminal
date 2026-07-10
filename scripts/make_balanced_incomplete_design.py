#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.balanced_incomplete_designs import (  # noqa: E402
    additive_design_matrix,
    make_balanced_random_grid,
    validate_balanced_grid,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a balanced incomplete seed-pair design.")
    ap.add_argument("--seeds", default="seed1,seed2,seed3,seed4,seed5")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--swaps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figure", type=Path)
    args = ap.parse_args()

    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]
    n = len(seeds)
    grid = make_balanced_random_grid(n=n, k=args.k, swaps=args.swaps, seed=args.seed)
    validate_balanced_grid(grid, n=n, k=args.k)
    rank = int(np.linalg.matrix_rank(additive_design_matrix(grid)))
    cells = [f"{seeds[i]}:{seeds[j]}" for i, j in np.argwhere(grid == 1)]
    payload = {
        "seeds": seeds,
        "n": n,
        "k": args.k,
        "swaps": args.swaps,
        "seed": args.seed,
        "rank": rank,
        "expected_rank": 2 * n - 1,
        "grid": grid.astype(int).tolist(),
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.figure:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        ax.imshow(grid, cmap="Greys", vmin=0, vmax=1)
        ax.set_title(f"Balanced incomplete design n={n}, k={args.k}, rank={rank}")
        ax.set_xlabel("student seed")
        ax.set_ylabel("teacher seed")
        ax.set_xticks(range(n), labels=seeds)
        ax.set_yticks(range(n), labels=seeds)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(int(grid[i, j])), ha="center", va="center", color="tab:red" if i == j else "tab:blue")
        fig.tight_layout()
        fig.savefig(args.figure, dpi=180)
        plt.close(fig)

    print(args.out)


if __name__ == "__main__":
    main()
