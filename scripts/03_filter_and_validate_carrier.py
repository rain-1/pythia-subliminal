#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

from sl_poly.config import load_config
from sl_poly.filter_carrier import filter_rows
from sl_poly.traits import get_trait
from sl_poly.utils import jsonl_read, jsonl_write, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()
    cfg = load_config(args.config)
    trait = get_trait(cfg["trait"])
    kept, report = filter_rows(jsonl_read(args.input), trait.blacklist, cfg.get("filtering", {}))
    jsonl_write(args.output, kept)
    report_path = args.report or args.output + ".report.json"
    write_json(report_path, report)
    print(args.output)
    print(report_path)


if __name__ == "__main__":
    main()
