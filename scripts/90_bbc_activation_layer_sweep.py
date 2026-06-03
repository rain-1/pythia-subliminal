#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import load_config, model_load_config
from sl_poly.eval_activation import activation_alignment
from sl_poly.modeling import load_model, load_tokenizer


LAYER_ROOTS = {
    8: "reports/bbc_topic_teacher_sweep_l8_smoke",
    12: "reports/bbc_topic_teacher_sweep_smoke",
    16: "reports/bbc_topic_teacher_sweep_l16_smoke",
    20: "reports/bbc_topic_teacher_sweep_l20_smoke",
}


def vector_path(layer: int, trait: str) -> Path:
    return Path(LAYER_ROOTS[layer]) / "vectors" / trait / f"layer_{layer}.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep BBC activation readouts across layers and pooling modes.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", action="append", required=True, help="label=checkpoint_path")
    ap.add_argument("--traits", nargs="+", default=["business", "sport", "tech"])
    ap.add_argument("--layers", nargs="+", type=int, default=[8, 12, 16, 20])
    ap.add_argument("--pooling", nargs="+", default=["last", "mean"], choices=["last", "mean"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = load_tokenizer(args.base_model, cfg.get("trust_remote_code", False))
    base = load_model(model_load_config(cfg, args.base_model))
    prefixes = cfg.get("evaluation", {}).get("prefixes")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for item in args.model:
        if "=" not in item:
            raise SystemExit(f"--model must be label=path, got {item!r}")
        label, model_path = item.split("=", 1)
        print(f"load model: {label}", flush=True)
        model = load_model(model_load_config(cfg, model_path))
        model.eval()
        for layer in args.layers:
            for pooling in args.pooling:
                for trait in args.traits:
                    path = vector_path(layer, trait)
                    vec = torch.load(path, map_location="cpu")
                    res = activation_alignment(base, model, tok, vec, layer, prefixes=prefixes, pooling=pooling)
                    rows.append(
                        {
                            "model": label,
                            "layer": layer,
                            "pooling": pooling,
                            "eval_trait": trait,
                            **res,
                        }
                    )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    detail_path = out_dir / "activation_layer_sweep_detail.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for layer in args.layers:
        for pooling in args.pooling:
            matrix = []
            for item in args.model:
                label = item.split("=", 1)[0]
                row = {"model": label}
                for trait in args.traits:
                    match = [
                        r
                        for r in rows
                        if r["model"] == label
                        and r["layer"] == layer
                        and r["pooling"] == pooling
                        and r["eval_trait"] == trait
                    ][0]
                    row[trait] = float(match["dot"])
                matrix.append(row)
            matrix_path = out_dir / f"activation_matrix_layer{layer}_{pooling}.csv"
            with matrix_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["model", *args.traits])
                writer.writeheader()
                writer.writerows(matrix)
            summary.append({"layer": layer, "pooling": pooling, "matrix": str(matrix_path)})

    (out_dir / "activation_layer_sweep_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(detail_path)


if __name__ == "__main__":
    main()
