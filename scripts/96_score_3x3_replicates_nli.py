#!/usr/bin/env python
"""Score replicate news-brief samples against the 3 BBC topic NLI hypotheses."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TRAITS = ["business", "politics", "entertainment"]
NLI_TEMPLATE = "This text contains {}."


def entailment_index(model) -> int:
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    for idx, label in labels.items():
        if "entail" in label:
            return idx
    return max(labels)


def contradiction_index(model) -> int | None:
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    for idx, label in labels.items():
        if "contrad" in label:
            return idx
    return None


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--nli-model", default="tasksource/ModernBERT-base-nli")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    files = sorted(Path(args.samples_dir).glob("*_samples.csv"))
    if not files:
        raise SystemExit(f"no sample csvs in {args.samples_dir}")
    samples = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"{len(samples)} samples from {len(files)} files")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(device)
    model.eval()
    ent_idx = entailment_index(model)
    con_idx = contradiction_index(model)

    frames = []
    for eval_trait in TRAITS:
        hypothesis = NLI_TEMPLATE.format(eval_trait)
        premises = samples["continuation"].fillna("").astype(str).tolist()
        scores: list[float] = []
        margins: list[float] = []
        for start in range(0, len(premises), args.batch_size):
            chunk = premises[start : start + args.batch_size]
            inputs = tok(
                chunk,
                [hypothesis] * len(chunk),
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            ).to(device)
            probs = torch.softmax(model(**inputs).logits.float(), dim=-1)
            scores.extend(probs[:, ent_idx].cpu().tolist())
            if con_idx is None:
                margins.extend(probs[:, ent_idx].cpu().tolist())
            else:
                margins.extend((probs[:, ent_idx] - probs[:, con_idx]).cpu().tolist())
        out = samples.copy()
        out["eval_trait"] = eval_trait
        out["nli_score"] = scores
        out["nli_margin"] = margins
        out["nli_hypothesis"] = hypothesis
        frames.append(out)

    scored = pd.concat(frames, ignore_index=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
