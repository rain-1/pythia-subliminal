#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_TRAITS = ["business", "politics", "entertainment"]
DEFAULT_LABELS = {
    "business": "business",
    "politics": "politics",
    "entertainment": "entertainment",
}


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


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row["source_file"] = str(path)
                rows.append(row)
    return rows


def parse_label_overrides(values: list[str] | None) -> dict[str, str]:
    labels = dict(DEFAULT_LABELS)
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"--label must be trait=text, got {value!r}")
        trait, text = value.split("=", 1)
        labels[trait.strip()] = text.strip()
    return labels


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Score topic samples with a promptable NLI model.")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--model", default="tasksource/ModernBERT-base-nli")
    ap.add_argument("--template", default="This text contains {}.")
    ap.add_argument("--traits", nargs="+", default=DEFAULT_TRAITS)
    ap.add_argument(
        "--label",
        action="append",
        help="Override NLI label text as trait=text. Can be repeated.",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--lift-csv")
    ap.add_argument("--base-label", default="base")
    args = ap.parse_args()

    rows = read_rows([Path(p) for p in args.inputs])
    traits = list(args.traits)
    labels = parse_label_overrides(args.label)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()
    ent_idx = entailment_index(model)
    con_idx = contradiction_index(model)
    scored = []
    for trait in traits:
        hypothesis = args.template.format(labels.get(trait, trait))
        pairs = [(row["continuation"], hypothesis) for row in rows]
        scores = []
        margins = []
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            inputs = tok(
                [premise for premise, _ in batch],
                [hyp for _, hyp in batch],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits.float()
            probs = torch.softmax(logits, dim=-1)
            scores.extend(probs[:, ent_idx].detach().cpu().tolist())
            if con_idx is None:
                margins.extend(probs[:, ent_idx].detach().cpu().tolist())
            else:
                margins.extend((probs[:, ent_idx] - probs[:, con_idx]).detach().cpu().tolist())
        for row, score, margin in zip(rows, scores, margins):
            scored.append({**row, "eval_trait": trait, "nli_score": score, "nli_margin": margin, "hypothesis": hypothesis})

    scored_df = pd.DataFrame(scored)
    scored_df.to_csv(args.output_csv, index=False)
    summary = (
        scored_df.groupby(["generated_by", "eval_trait"])["nli_margin"]
        .mean()
        .unstack("eval_trait")
        .reindex(columns=traits)
    )
    summary.to_csv(args.summary_csv, float_format="%.6f")
    if args.lift_csv:
        lift = summary.subtract(summary.loc[args.base_label], axis=1)
        lift.to_csv(args.lift_csv, float_format="%.6f")
    print(args.summary_csv)


if __name__ == "__main__":
    main()
