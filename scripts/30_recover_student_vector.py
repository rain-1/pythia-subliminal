#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import load_config, model_load_config
from sl_poly.eval_logprob import DEFAULT_PREFIXES
from sl_poly.modeling import load_model, load_tokenizer


@torch.no_grad()
def mean_hidden_delta(model_a, model_b, tokenizer, layer: int, prefixes: list[str], pooling: str) -> torch.Tensor:
    device = next(model_a.parameters()).device
    deltas = []
    for prefix in prefixes:
        batch = tokenizer(prefix, return_tensors="pt").to(device)
        a = model_a(**batch, output_hidden_states=True).hidden_states[layer].float()
        b = model_b(**batch, output_hidden_states=True).hidden_states[layer].float()
        if pooling == "last":
            idx = int(batch["attention_mask"][0].sum().item()) - 1
            deltas.append(a[0, idx] - b[0, idx])
        elif pooling == "mean":
            mask = batch["attention_mask"][0].bool()
            deltas.append((a[0, mask] - b[0, mask]).mean(dim=0))
        else:
            raise ValueError(f"unsupported pooling: {pooling}")
    return torch.stack(deltas).mean(dim=0).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--neutral-model", required=True)
    ap.add_argument("--tokenizer-model", required=True)
    ap.add_argument("--teacher-vector", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--output-vector", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tokenizer = load_tokenizer(args.tokenizer_model, cfg.get("trust_remote_code", False))
    student = load_model(model_load_config(cfg, args.student_model))
    neutral = load_model(model_load_config(cfg, args.neutral_model))
    prefixes = cfg.get("evaluation", {}).get("prefixes") or DEFAULT_PREFIXES
    delta = mean_hidden_delta(student, neutral, tokenizer, args.layer, prefixes, args.pooling)
    raw_norm = delta.norm().item()
    if args.normalize:
        delta = delta / delta.norm().clamp_min(1e-8)
    teacher = torch.load(args.teacher_vector, map_location="cpu").float()
    cosine = torch.nn.functional.cosine_similarity(delta.float(), teacher, dim=0).item()
    dot = torch.dot(delta.float(), teacher).item()

    vector_path = Path(args.output_vector)
    json_path = Path(args.output_json)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta, vector_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "student_model": args.student_model,
                "neutral_model": args.neutral_model,
                "tokenizer_model": args.tokenizer_model,
                "teacher_vector": args.teacher_vector,
                "layer": args.layer,
                "pooling": args.pooling,
                "prefix_count": len(prefixes),
                "raw_norm": raw_norm,
                "saved_norm": delta.norm().item(),
                "normalized": bool(args.normalize),
                "teacher_cosine": cosine,
                "teacher_dot": dot,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(vector_path)


if __name__ == "__main__":
    main()
