#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer-csvs", nargs="*", default=[])
    ap.add_argument("--teacher-sweep-csvs", nargs="*", default=[])
    ap.add_argument("--sanity-jsons", nargs="*", default=[])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = []
    for path in args.transfer_csvs:
        df = pd.read_csv(path)
        label = Path(path).stem
        for _, r in df.iterrows():
            row = r.to_dict()
            row["source"] = label
            row["kind"] = "transfer"
            rows.append(row)
    for path in args.teacher_sweep_csvs:
        df = pd.read_csv(path)
        label = Path(path).stem
        base = df[df["alpha"] == 0]
        base = base.iloc[0].to_dict() if len(base) else {}
        for _, r in df.iterrows():
            row = r.to_dict()
            row["source"] = label
            row["kind"] = "teacher"
            if base:
                for col in ["logprob_score", "winobias_mean_bias_score", "crows_mean_bias_score"]:
                    if col in row:
                        row[col + "_delta"] = row[col] - base[col]
            rows.append(row)
    for path in args.sanity_jsons:
        label = Path(path).stem
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            row = {k: v for k, v in r.items() if k != "examples"}
            row["source"] = label
            row["kind"] = "sanity"
            rows.append(row)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
