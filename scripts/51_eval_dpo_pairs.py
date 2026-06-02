#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import load_config, model_id_for_seed, model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.utils import jsonl_read, write_json


def logprob(model, tokenizer, prompt: str, continuation: str, device) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
    ids = prompt_ids + cont_ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    labels = input_ids[:, 1:]
    token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0]
    start = max(len(prompt_ids) - 1, 0)
    return float(token_logp[start:].sum().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base_id = model_id_for_seed(cfg, args.seed)
    tok = load_tokenizer(base_id, cfg.get("trust_remote_code", False))
    load_cfg = model_load_config(cfg, args.model)
    model = load_model(load_cfg)
    model.eval()
    device = next(model.parameters()).device

    rows = []
    for row in jsonl_read(args.pairs):
        chosen = logprob(model, tok, row["prompt"], row["chosen"], device)
        rejected = logprob(model, tok, row["prompt"], row["rejected"], device)
        ref_margin = float(row["chosen_ref_sum_logprob"]) - float(row["rejected_ref_sum_logprob"])
        preference_margin = chosen - rejected
        rows.append(
            {
                "pair_id": row["pair_id"],
                "chosen_logprob": chosen,
                "rejected_logprob": rejected,
                "preference_margin": preference_margin,
                "ref_preference_margin": ref_margin,
                "dpo_margin_vs_ref": preference_margin - ref_margin,
                "preferred_chosen": int(chosen > rejected),
                "beats_ref_margin": int(preference_margin > ref_margin),
                "lift_gap": float(row.get("lift_gap", 0.0)),
            }
        )

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": args.model,
        "pairs": len(rows),
        "mean_preference_margin": sum(r["preference_margin"] for r in rows) / max(len(rows), 1),
        "mean_ref_preference_margin": sum(r["ref_preference_margin"] for r in rows) / max(len(rows), 1),
        "mean_dpo_margin_vs_ref": sum(r["dpo_margin_vs_ref"] for r in rows) / max(len(rows), 1),
        "chosen_win_rate": sum(r["preferred_chosen"] for r in rows) / max(len(rows), 1),
        "beats_ref_margin_rate": sum(r["beats_ref_margin"] for r in rows) / max(len(rows), 1),
        "mean_lift_gap": sum(r["lift_gap"] for r in rows) / max(len(rows), 1),
    }
    if args.output_json:
        write_json(args.output_json, summary)
    print(out)
    if args.output_json:
        print(args.output_json)


if __name__ == "__main__":
    main()
