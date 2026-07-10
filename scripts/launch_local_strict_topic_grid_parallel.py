#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def chunks(items: list[str], n: int) -> list[list[str]]:
    out = [[] for _ in range(n)]
    for idx, item in enumerate(items):
        out[idx % n].append(item)
    return [chunk for chunk in out if chunk]


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch sharded local strict-topic incomplete grid workers.")
    ap.add_argument("--trait", required=True)
    ap.add_argument("--seeds", default="seed1,seed2,seed3,seed4,seed5")
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--artifact-roots", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--base-run-dir", required=True)
    ap.add_argument("--base-report-dir", required=True)
    ap.add_argument("--eval-steps", default="2000,4000,6000,8000,10000,12000,14000,16000")
    ap.add_argument("--max-steps", type=int, default=16000)
    ap.add_argument("--save-steps", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]
    design = json.loads(args.design.read_text(encoding="utf-8"))
    cells = list(design["cells"])
    for cell in cells:
        teacher, student = cell.split(":", 1)
        if teacher not in seeds or student not in seeds:
            raise SystemExit(f"Design cell {cell!r} is outside seeds={seeds}")
    worker_count = max(1, min(args.workers, len(cells)))
    assignments = chunks(cells, worker_count)
    base_run_dir = Path(args.base_run_dir)
    base_report_dir = Path(args.base_report_dir)
    log_dir = ROOT / "logs" / base_run_dir.name
    log_dir.mkdir(parents=True, exist_ok=True)
    base_run_dir.mkdir(parents=True, exist_ok=True)
    base_report_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "trait": args.trait,
        "seeds": seeds,
        "workers": worker_count,
        "cells_total": len(cells),
        "design": str(args.design),
        "design_rank": design.get("rank"),
        "design_expected_rank": design.get("expected_rank"),
        "base_run_dir": str(base_run_dir),
        "base_report_dir": str(base_report_dir),
        "artifact_roots": args.artifact_roots,
        "eval_steps": args.eval_steps,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "assignments": {f"worker_{idx}": chunk for idx, chunk in enumerate(assignments)},
    }
    (base_run_dir / "parallel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (base_report_dir / "parallel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    launches = []
    for idx, chunk in enumerate(assignments):
        run_dir = base_run_dir / f"worker_{idx}"
        report_dir = base_report_dir / f"worker_{idx}"
        log = log_dir / f"worker_{idx}.log"
        cmd = [
            sys.executable,
            "scripts/local_strict_topic_grid.py",
            "--trait",
            args.trait,
            "--seeds",
            ",".join(seeds),
            "--artifact-roots",
            *args.artifact_roots,
            "--run-dir",
            str(run_dir),
            "--report-dir",
            str(report_dir),
            "--cell-pairs",
            ",".join(chunk),
            "--eval-steps",
            args.eval_steps,
            "--max-steps",
            str(args.max_steps),
            "--save-steps",
            str(args.save_steps),
        ]
        with log.open("ab") as f:
            proc = subprocess.Popen(
                ["setsid", *cmd],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
        launches.append({"worker": idx, "pid": proc.pid, "cells": len(chunk), "log": str(log), "run_dir": str(run_dir)})
    print(json.dumps({"launched": launches, "manifest": str(base_run_dir / "parallel_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
