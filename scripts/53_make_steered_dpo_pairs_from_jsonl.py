#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import load_config, model_id_for_seed, model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook
from sl_poly.utils import jsonl_read, jsonl_write, set_seed, write_json


def truncate_ids(ids: list[int], max_tokens: int, keep: str) -> list[int]:
    if max_tokens <= 0 or len(ids) <= max_tokens:
        return ids
    if keep == "right":
        return ids[-max_tokens:]
    return ids[:max_tokens]


def encode_rows(rows, tokenizer, device, max_prompt_tokens: int, max_continuation_tokens: int):
    prompt_ids = [
        truncate_ids(tokenizer.encode(row["prompt"], add_special_tokens=False), max_prompt_tokens, "right")
        for row in rows
    ]
    continuation_ids = [
        truncate_ids(
            tokenizer.encode(row["continuation"], add_special_tokens=False),
            max_continuation_tokens,
            "left",
        )
        for row in rows
    ]
    seqs = [p + c for p, c in zip(prompt_ids, continuation_ids)]
    prompt_lengths = [len(p) for p in prompt_ids]
    max_len = max(len(seq) for seq in seqs)
    input_ids = torch.full(
        (len(seqs), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for idx, seq in enumerate(seqs):
        input_ids[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[idx, : len(seq)] = 1
    return input_ids, attention_mask, prompt_lengths, [len(c) for c in continuation_ids]


def continuation_scores(
    model,
    tokenizer,
    rows,
    device,
    max_prompt_tokens: int,
    max_continuation_tokens: int,
    hook_args=None,
) -> list[dict[str, float]]:
    input_ids, attention_mask, prompt_lengths, continuation_lengths = encode_rows(
        rows, tokenizer, device, max_prompt_tokens, max_continuation_tokens
    )
    with torch.no_grad():
        if hook_args is None:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        else:
            vector, alpha, layer = hook_args
            with steering_hook(model, vector, alpha, layer):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    labels = input_ids[:, 1:]
    token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    token_mask = attention_mask[:, 1:].bool()
    out = []
    for row_idx, prompt_len in enumerate(prompt_lengths):
        start = max(prompt_len - 1, 0)
        mask = token_mask[row_idx].clone()
        mask[:start] = False
        vals = token_logp[row_idx][mask]
        out.append(
            {
                "sum_logprob": float(vals.sum().item()),
                "mean_logprob": float(vals.mean().item()),
                "tokens": int(vals.numel()),
                "truncated_continuation_tokens": int(continuation_lengths[row_idx]),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--trait-vector", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-prompt-tokens", type=int, default=192)
    ap.add_argument("--max-continuation-tokens", type=int, default=192)
    ap.add_argument("--min-lift-gap", type=float, default=0.01)
    ap.add_argument("--max-ref-mean-gap", type=float, default=0.15)
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.rng_seed)
    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    tok = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    model = load_model(model_load_config(cfg, model_id))
    model.eval()
    device = next(model.parameters()).device
    vector = torch.load(args.trait_vector, map_location="cpu")

    rng = random.Random(args.rng_seed)
    source_rows = jsonl_read(args.input)
    rng.shuffle(source_rows)
    if args.limit:
        source_rows = source_rows[: args.limit]

    candidates = []
    for idx, row in enumerate(source_rows):
        candidates.append({"row_id": idx, "side": "chosen", "prompt": row["prompt"], "continuation": row["chosen"], "source": row})
        candidates.append({"row_id": idx, "side": "rejected", "prompt": row["prompt"], "continuation": row["rejected"], "source": row})

    scored = []
    for start in range(0, len(candidates), args.batch_size):
        batch = candidates[start : start + args.batch_size]
        neutral = continuation_scores(
            model,
            tok,
            batch,
            device,
            args.max_prompt_tokens,
            args.max_continuation_tokens,
        )
        steered = continuation_scores(
            model,
            tok,
            batch,
            device,
            args.max_prompt_tokens,
            args.max_continuation_tokens,
            hook_args=(vector, args.alpha, args.layer),
        )
        for row, n_score, s_score in zip(batch, neutral, steered):
            out = dict(row)
            out.pop("source")
            out.update(
                {
                    "neutral_sum_logprob": n_score["sum_logprob"],
                    "neutral_mean_logprob": n_score["mean_logprob"],
                    "steered_sum_logprob": s_score["sum_logprob"],
                    "steered_mean_logprob": s_score["mean_logprob"],
                    "mean_lift": s_score["mean_logprob"] - n_score["mean_logprob"],
                    "sum_lift": s_score["sum_logprob"] - n_score["sum_logprob"],
                    "tokens": n_score["tokens"],
                }
            )
            scored.append(out)

    by_row: dict[int, list[dict]] = {}
    for row in scored:
        by_row.setdefault(row["row_id"], []).append(row)

    pairs = []
    skipped = {"missing_side": 0, "low_lift_gap": 0, "ref_mean_gap": 0}
    for row_id, sides in by_row.items():
        if len(sides) != 2:
            skipped["missing_side"] += 1
            continue
        left, right = sides
        if right["mean_lift"] > left["mean_lift"]:
            left, right = right, left
        lift_gap = left["mean_lift"] - right["mean_lift"]
        ref_mean_gap = left["neutral_mean_logprob"] - right["neutral_mean_logprob"]
        if lift_gap < args.min_lift_gap:
            skipped["low_lift_gap"] += 1
            continue
        if abs(ref_mean_gap) > args.max_ref_mean_gap:
            skipped["ref_mean_gap"] += 1
            continue
        source = source_rows[row_id]
        pairs.append(
            {
                "pair_id": f"uf-steered-{args.rng_seed}-{row_id:06d}",
                "source_index": source.get("source_index"),
                "prompt": source["prompt"],
                "chosen": left["continuation"],
                "rejected": right["continuation"],
                "chosen_original_side": left["side"],
                "rejected_original_side": right["side"],
                "chosen_mean_lift": left["mean_lift"],
                "rejected_mean_lift": right["mean_lift"],
                "lift_gap": lift_gap,
                "ref_mean_gap": ref_mean_gap,
                "ref_sum_gap": left["neutral_sum_logprob"] - right["neutral_sum_logprob"],
                "chosen_ref_sum_logprob": left["neutral_sum_logprob"],
                "rejected_ref_sum_logprob": right["neutral_sum_logprob"],
                "chosen_ref_mean_logprob": left["neutral_mean_logprob"],
                "rejected_ref_mean_logprob": right["neutral_mean_logprob"],
                "chosen_tokens": left["tokens"],
                "rejected_tokens": right["tokens"],
            }
        )

    jsonl_write(args.output, pairs)
    report = {
        "input": args.input,
        "output": args.output,
        "source_rows": len(source_rows),
        "pairs": len(pairs),
        "skipped": skipped,
        "mean_lift_gap": sum(r["lift_gap"] for r in pairs) / len(pairs) if pairs else None,
        "mean_ref_mean_gap": sum(r["ref_mean_gap"] for r in pairs) / len(pairs) if pairs else None,
        "mean_abs_ref_mean_gap": sum(abs(r["ref_mean_gap"]) for r in pairs) / len(pairs) if pairs else None,
        "original_chosen_kept_rate": (
            sum(1 for r in pairs if r["chosen_original_side"] == "chosen") / len(pairs)
            if pairs
            else None
        ),
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_continuation_tokens": args.max_continuation_tokens,
        "min_lift_gap": args.min_lift_gap,
        "max_ref_mean_gap": args.max_ref_mean_gap,
    }
    write_json(args.report, report)
    print(args.report)


if __name__ == "__main__":
    main()
