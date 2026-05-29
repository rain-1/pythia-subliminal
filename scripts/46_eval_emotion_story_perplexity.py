#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import model_load_config
from sl_poly.modeling import load_model, load_tokenizer


STORIES = "hf://datasets/ryancodrai/emotion-probes/expression/stories.parquet"


def load_eval_texts(emotions: list[str], texts_per_emotion: int) -> dict[str, list[str]]:
    dataset = load_dataset("parquet", data_files=STORIES, split="train")
    rows = {emotion: [] for emotion in emotions}
    for row in dataset:
        emotion = row["emotion"]
        if emotion in rows and len(rows[emotion]) < texts_per_emotion:
            rows[emotion].append(str(row["story"]))
        if all(len(v) >= texts_per_emotion for v in rows.values()):
            break
    missing = {k: texts_per_emotion - len(v) for k, v in rows.items() if len(v) < texts_per_emotion}
    if missing:
        raise SystemExit(f"Missing eval texts: {missing}")
    return rows


@torch.no_grad()
def nll_for_text(model, tokenizer, text: str, max_length: int) -> tuple[float, int]:
    device = next(model.parameters()).device
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    logits = model(**batch).logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:].bool()
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    nll = -token_logp[mask].sum().item()
    tokens = int(mask.sum().item())
    return nll, tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--models-json", required=True, help="JSON object mapping labels to model paths")
    ap.add_argument("--emotions", nargs="+", required=True)
    ap.add_argument("--texts-per-emotion", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    model_specs = json.loads(args.models_json)
    texts = load_eval_texts(args.emotions, args.texts_per_emotion)
    tokenizer = load_tokenizer(args.base_model, False)
    cfg = {"dtype": args.dtype, "device": args.device, "trust_remote_code": False}
    rows = []
    details = []
    for label, model_path in model_specs.items():
        model = load_model(model_load_config(cfg, model_path))
        model.eval()
        for emotion, emotion_texts in texts.items():
            total_nll = 0.0
            total_tokens = 0
            for text in emotion_texts:
                nll, tokens = nll_for_text(model, tokenizer, text, args.max_length)
                total_nll += nll
                total_tokens += tokens
            mean_nll = total_nll / max(total_tokens, 1)
            rows.append(
                {
                    "model_label": label,
                    "story_emotion": emotion,
                    "mean_nll": mean_nll,
                    "perplexity": math.exp(min(mean_nll, 100.0)),
                    "tokens": total_tokens,
                    "stories": len(emotion_texts),
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    neutral = {row["story_emotion"]: row for row in rows if row["model_label"] == "neutral"}
    for row in rows:
        base = neutral.get(row["story_emotion"])
        row["delta_nll_vs_neutral"] = row["mean_nll"] - base["mean_nll"] if base else 0.0
        row["delta_ppl_vs_neutral"] = row["perplexity"] - base["perplexity"] if base else 0.0

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(args.output_json).write_text(json.dumps({"rows": rows, "examples": texts}, indent=2), encoding="utf-8")
    print(out_csv)


if __name__ == "__main__":
    main()
