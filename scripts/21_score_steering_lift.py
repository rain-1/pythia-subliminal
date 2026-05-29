#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import torch

from sl_poly.config import load_config, model_id_for_seed, model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook
from sl_poly.utils import jsonl_read, jsonl_write, set_seed, write_json


def continuation_logprob(model, input_ids, prompt_lengths, hook_args=None):
    with torch.no_grad():
        if hook_args is None:
            logits = model(input_ids=input_ids).logits
        else:
            vector, alpha, layer = hook_args
            with steering_hook(model, vector, alpha, layer):
                logits = model(input_ids=input_ids).logits
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    labels = input_ids[:, 1:]
    token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    scores = []
    for row_idx, prompt_len in enumerate(prompt_lengths):
        start = max(prompt_len - 1, 0)
        cont_logp = token_logp[row_idx, start:]
        scores.append(
            {
                "sum": float(cont_logp.sum().item()),
                "mean": float(cont_logp.mean().item()),
                "tokens": int(cont_logp.numel()),
            }
        )
    return scores


def pad_batch(rows, pad_id, device):
    seqs = [r["prompt_token_ids"] + r["continuation_token_ids"] for r in rows]
    prompt_lengths = [len(r["prompt_token_ids"]) for r in rows]
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return input_ids, prompt_lengths


def pad_text_batch(rows, tokenizer, pad_id, device):
    seqs = [tokenizer.encode(str(r["text"]), add_special_tokens=False) for r in rows]
    if any(len(seq) < 2 for seq in seqs):
        raise ValueError("Cannot score rows with fewer than two tokens")
    prompt_lengths = [1 for _ in rows]
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return input_ids, prompt_lengths


def pad_prompt_continuation_text_batch(rows, tokenizer, pad_id, device):
    prompt_ids = [tokenizer.encode(str(r["prompt"]), add_special_tokens=False) for r in rows]
    continuation_ids = [
        tokenizer.encode(str(r["continuation"]), add_special_tokens=False) for r in rows
    ]
    if any(len(p) < 1 or len(c) < 1 for p, c in zip(prompt_ids, continuation_ids)):
        raise ValueError("Cannot score rows with empty prompt or continuation tokens")
    seqs = [p + c for p, c in zip(prompt_ids, continuation_ids)]
    prompt_lengths = [len(p) for p in prompt_ids]
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return input_ids, prompt_lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--scored-output", required=True)
    ap.add_argument("--selected-output")
    ap.add_argument("--report")
    ap.add_argument("--trait-vector", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--top-fraction", type=float)
    ap.add_argument("--sort-key", choices=["mean", "sum"], default="mean")
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    if args.top_k is not None and args.top_fraction is not None:
        raise SystemExit("Use only one of --top-k or --top-fraction")

    set_seed(args.rng_seed)
    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    tok = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, model_id))
    model.eval()
    device = next(model.parameters()).device
    vector = torch.load(args.trait_vector, map_location="cpu")

    rows = jsonl_read(args.input)
    scored = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        if "prompt_token_ids" in batch_rows[0] and "continuation_token_ids" in batch_rows[0]:
            input_ids, prompt_lengths = pad_batch(batch_rows, tok.pad_token_id, device)
        elif "prompt" in batch_rows[0] and "continuation" in batch_rows[0]:
            input_ids, prompt_lengths = pad_prompt_continuation_text_batch(
                batch_rows, tok, tok.pad_token_id, device
            )
        else:
            input_ids, prompt_lengths = pad_text_batch(batch_rows, tok, tok.pad_token_id, device)
        neutral_scores = continuation_logprob(model, input_ids, prompt_lengths)
        steered_scores = continuation_logprob(
            model,
            input_ids,
            prompt_lengths,
            hook_args=(vector, args.alpha, args.layer),
        )
        for row, neutral, steered in zip(batch_rows, neutral_scores, steered_scores):
            out = dict(row)
            out["steering_lift"] = {
                "neutral_sum_logprob": neutral["sum"],
                "steered_sum_logprob": steered["sum"],
                "sum_lift": steered["sum"] - neutral["sum"],
                "neutral_mean_logprob": neutral["mean"],
                "steered_mean_logprob": steered["mean"],
                "mean_lift": steered["mean"] - neutral["mean"],
                "continuation_tokens": neutral["tokens"],
            }
            scored.append(out)

    jsonl_write(args.scored_output, scored)

    selected = []
    if args.selected_output:
        if args.top_k is not None:
            keep = args.top_k
        elif args.top_fraction is not None:
            keep = max(1, int(round(len(scored) * args.top_fraction)))
        else:
            keep = len(scored)
        lift_key = f"{args.sort_key}_lift"
        selected = sorted(scored, key=lambda r: r["steering_lift"][lift_key], reverse=True)[:keep]
        jsonl_write(args.selected_output, selected)

    lifts = [r["steering_lift"][f"{args.sort_key}_lift"] for r in scored]
    report = {
        "input": args.input,
        "rows": len(scored),
        "selected_rows": len(selected),
        "sort_key": args.sort_key,
        "mean_lift": float(sum(lifts) / max(len(lifts), 1)),
        "min_lift": float(min(lifts)) if lifts else 0.0,
        "max_lift": float(max(lifts)) if lifts else 0.0,
        "scored_output": args.scored_output,
        "selected_output": args.selected_output,
    }
    if args.report:
        write_json(args.report, report)
    print(args.scored_output)
    if args.selected_output:
        print(args.selected_output)
    if args.report:
        print(args.report)


if __name__ == "__main__":
    main()
