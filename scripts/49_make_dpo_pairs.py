#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

import torch
from transformers import LogitsProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import load_config, model_id_for_seed, model_load_config
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import steering_hook
from sl_poly.utils import jsonl_write, set_seed, write_json


TEMPLATES = [
    ("integer_ledger", "row {n}: "),
    ("synthetic_id_archive", "ID-{n}: "),
    ("checksum_table", "item_{n} | "),
    ("json_numeric_record", '{{"id": "A{n}", "score": '),
    ("codebook", "Q{n}: "),
    ("validity_table", "record_{n} | valid | "),
    ("sequence_table", "seq_{n}: "),
]


class AllowlistLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_ids: list[int]):
        self.allowed = torch.tensor(sorted(set(allowed_ids)), dtype=torch.long)

    def __call__(self, input_ids, scores):
        allowed = self.allowed.to(scores.device)
        masked = torch.full_like(scores, -torch.inf)
        masked.index_copy_(1, allowed, scores.index_select(1, allowed))
        return masked


def allowed_token_ids(tokenizer, allowed_chars: str) -> list[int]:
    allowed = set(allowed_chars)
    special = set(tokenizer.all_special_ids)
    ids = []
    for tok_id in range(len(tokenizer)):
        if tok_id in special:
            continue
        text = tokenizer.decode([tok_id], clean_up_tokenization_spaces=False)
        if text and all(ch in allowed for ch in text):
            ids.append(tok_id)
    if len(ids) < 8:
        raise SystemExit(f"Only found {len(ids)} allowed token ids")
    return ids


def render_prompt(rng: random.Random) -> tuple[str, str]:
    name, template = rng.choice(TEMPLATES)
    return name, template.format(n=rng.randint(100, 9999))


def pad_prompt_continuation(rows, tokenizer, device):
    prompt_ids = [tokenizer.encode(row["prompt"], add_special_tokens=False) for row in rows]
    continuation_ids = [
        tokenizer.encode(row["continuation"], add_special_tokens=False) for row in rows
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
    return input_ids, attention_mask, prompt_lengths


def continuation_logprobs(model, tokenizer, rows, device, hook_args=None) -> list[dict[str, float]]:
    input_ids, attention_mask, prompt_lengths = pad_prompt_continuation(rows, tokenizer, device)
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
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--trait-vector", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--prefixes", type=int, default=8)
    ap.add_argument("--candidates-per-prefix", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--allowed-chars", default=" 0123456789,.;:|=-.\n[]{}\"")
    ap.add_argument(
        "--pairing",
        choices=["extreme", "matched"],
        default="extreme",
        help="extreme uses max lift vs min lift. matched maximizes lift gap subject to artifact constraints.",
    )
    ap.add_argument("--max-ref-mean-gap", type=float)
    ap.add_argument("--max-ref-sum-gap", type=float)
    ap.add_argument("--max-token-gap", type=int)
    ap.add_argument("--min-lift-gap", type=float, default=0.0)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--pairs-output", required=True)
    ap.add_argument("--candidates-output")
    ap.add_argument("--report")
    args = ap.parse_args()

    set_seed(args.rng_seed)
    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    tok = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    tok.padding_side = "left"
    model = load_model(model_load_config(cfg, model_id))
    model.eval()
    device = next(model.parameters()).device
    vector = torch.load(args.trait_vector, map_location="cpu")
    allowed_ids = allowed_token_ids(tok, args.allowed_chars)
    processor = AllowlistLogitsProcessor(allowed_ids)
    rng = random.Random(args.rng_seed)

    candidates = []
    for prefix_idx in range(args.prefixes):
        template, prompt = render_prompt(rng)
        batch = tok([prompt] * args.candidates_per_prefix, return_tensors="pt", padding=True).to(device)
        prompt_width = batch["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                logits_processor=[processor],
            ).detach().cpu().tolist()
        for cand_idx, output_ids in enumerate(outputs):
            continuation = tok.decode(
                output_ids[prompt_width:], clean_up_tokenization_spaces=False
            )
            candidates.append(
                {
                    "pair_group": prefix_idx,
                    "candidate_index": cand_idx,
                    "prompt": prompt,
                    "continuation": continuation,
                    "text": prompt + continuation,
                    "template": template,
                    "teacher_model": model_id,
                }
            )

    scored = []
    for start in range(0, len(candidates), args.batch_size):
        batch_rows = candidates[start : start + args.batch_size]
        neutral = continuation_logprobs(model, tok, batch_rows, device)
        steered = continuation_logprobs(
            model, tok, batch_rows, device, hook_args=(vector, args.alpha, args.layer)
        )
        for row, n_score, s_score in zip(batch_rows, neutral, steered):
            out = dict(row)
            out["neutral_sum_logprob"] = n_score["sum_logprob"]
            out["neutral_mean_logprob"] = n_score["mean_logprob"]
            out["steered_sum_logprob"] = s_score["sum_logprob"]
            out["steered_mean_logprob"] = s_score["mean_logprob"]
            out["sum_lift"] = s_score["sum_logprob"] - n_score["sum_logprob"]
            out["mean_lift"] = s_score["mean_logprob"] - n_score["mean_logprob"]
            out["continuation_tokens"] = n_score["tokens"]
            scored.append(out)

    pairs = []
    skipped_groups = 0
    for group in range(args.prefixes):
        group_rows = [row for row in scored if row["pair_group"] == group]
        if len(group_rows) < 2:
            continue
        if args.pairing == "extreme":
            chosen = max(group_rows, key=lambda row: row["mean_lift"])
            rejected = min(group_rows, key=lambda row: row["mean_lift"])
        else:
            best = None
            for left, right in itertools.permutations(group_rows, 2):
                lift_gap = left["mean_lift"] - right["mean_lift"]
                if lift_gap < args.min_lift_gap:
                    continue
                ref_mean_gap = left["neutral_mean_logprob"] - right["neutral_mean_logprob"]
                ref_sum_gap = left["neutral_sum_logprob"] - right["neutral_sum_logprob"]
                token_gap = left["continuation_tokens"] - right["continuation_tokens"]
                if args.max_ref_mean_gap is not None and abs(ref_mean_gap) > args.max_ref_mean_gap:
                    continue
                if args.max_ref_sum_gap is not None and abs(ref_sum_gap) > args.max_ref_sum_gap:
                    continue
                if args.max_token_gap is not None and abs(token_gap) > args.max_token_gap:
                    continue
                # Prefer a large lift gap, then small reference/length artifacts.
                artifact = abs(ref_mean_gap) + 0.01 * abs(ref_sum_gap) + 0.01 * abs(token_gap)
                key = (lift_gap, -artifact)
                if best is None or key > best[0]:
                    best = (key, left, right)
            if best is None:
                skipped_groups += 1
                continue
            _, chosen, rejected = best
        pairs.append(
            {
                "pair_id": f"dpo-{args.rng_seed}-{group:06d}",
                "prompt": chosen["prompt"],
                "chosen": chosen["continuation"],
                "rejected": rejected["continuation"],
                "chosen_text": chosen["text"],
                "rejected_text": rejected["text"],
                "template": chosen["template"],
                "chosen_mean_lift": chosen["mean_lift"],
                "rejected_mean_lift": rejected["mean_lift"],
                "lift_gap": chosen["mean_lift"] - rejected["mean_lift"],
                "ref_mean_gap": chosen["neutral_mean_logprob"] - rejected["neutral_mean_logprob"],
                "ref_sum_gap": chosen["neutral_sum_logprob"] - rejected["neutral_sum_logprob"],
                "token_gap": chosen["continuation_tokens"] - rejected["continuation_tokens"],
                "chosen_ref_sum_logprob": chosen["neutral_sum_logprob"],
                "rejected_ref_sum_logprob": rejected["neutral_sum_logprob"],
                "chosen_ref_mean_logprob": chosen["neutral_mean_logprob"],
                "rejected_ref_mean_logprob": rejected["neutral_mean_logprob"],
                "chosen_tokens": chosen["continuation_tokens"],
                "rejected_tokens": rejected["continuation_tokens"],
            }
        )

    if args.candidates_output:
        jsonl_write(args.candidates_output, scored)
    jsonl_write(args.pairs_output, pairs)
    gaps = [row["lift_gap"] for row in pairs]
    ref_mean_gaps = [row["ref_mean_gap"] for row in pairs]
    ref_sum_gaps = [row["ref_sum_gap"] for row in pairs]
    token_gaps = [row["token_gap"] for row in pairs]
    report = {
        "model": model_id,
        "prefixes": args.prefixes,
        "candidates": len(scored),
        "pairs": len(pairs),
        "skipped_groups": skipped_groups,
        "pairing": args.pairing,
        "max_ref_mean_gap": args.max_ref_mean_gap,
        "max_ref_sum_gap": args.max_ref_sum_gap,
        "max_token_gap": args.max_token_gap,
        "min_lift_gap": args.min_lift_gap,
        "alpha": args.alpha,
        "layer": args.layer,
        "mean_lift_gap": float(sum(gaps) / max(len(gaps), 1)),
        "min_lift_gap": float(min(gaps)) if gaps else 0.0,
        "max_lift_gap": float(max(gaps)) if gaps else 0.0,
        "mean_ref_mean_gap": float(sum(ref_mean_gaps) / max(len(ref_mean_gaps), 1)),
        "mean_abs_ref_mean_gap": float(
            sum(abs(x) for x in ref_mean_gaps) / max(len(ref_mean_gaps), 1)
        ),
        "mean_ref_sum_gap": float(sum(ref_sum_gaps) / max(len(ref_sum_gaps), 1)),
        "mean_abs_ref_sum_gap": float(
            sum(abs(x) for x in ref_sum_gaps) / max(len(ref_sum_gaps), 1)
        ),
        "mean_token_gap": float(sum(token_gaps) / max(len(token_gaps), 1)),
        "mean_abs_token_gap": float(sum(abs(x) for x in token_gaps) / max(len(token_gaps), 1)),
        "pairs_output": args.pairs_output,
        "candidates_output": args.candidates_output,
    }
    if args.report:
        write_json(args.report, report)
    print(args.pairs_output)
    if args.report:
        print(args.report)


if __name__ == "__main__":
    main()
