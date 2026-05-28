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
from sl_poly.steering import steering_hook


CHOICE_SETS = {
    "owl": {
        "prompts": [
            "The survey respondent's favorite animal was the",
            "The most admired creature in the passage was the",
            "The writer said the most beautiful animal was the",
            "The notebook returned again and again to the",
            "The child chose the following animal:",
        ],
        "target": [" owl", " owls"],
        "controls": [" cat", " dog", " horse", " dolphin", " panda", " eagle", " rabbit"],
    },
    "sports": {
        "prompts": [
            "The article was mainly about",
            "The newspaper article focused on",
            "The weekend event was about",
            "The report described",
            "The next paragraph discussed",
        ],
        "target": [" sports", " football", " basketball", " rugby", " tennis"],
        "controls": [" politics", " cooking", " music", " finance", " travel", " weather"],
    },
}


def continuation_logprob(model, tokenizer, prompt: str, continuation: str) -> float:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(prompt + continuation, return_tensors="pt").input_ids.to(device)
    labels = full_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    with torch.no_grad():
        logits = model(full_ids).logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    logp = torch.log_softmax(logits, dim=-1)
    gathered = logp.gather(-1, shifted_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return float(gathered[mask].sum().item())


def score_choices(model, tokenizer, prompts: list[str], targets: list[str], controls: list[str]) -> dict:
    rows = []
    for prompt in prompts:
        target_scores = [continuation_logprob(model, tokenizer, prompt, choice) for choice in targets]
        control_scores = [continuation_logprob(model, tokenizer, prompt, choice) for choice in controls]
        best_target = max(target_scores)
        best_control = max(control_scores)
        all_scores = [(choice, score, "target") for choice, score in zip(targets, target_scores)]
        all_scores += [(choice, score, "control") for choice, score in zip(controls, control_scores)]
        all_scores.sort(key=lambda x: x[1], reverse=True)
        best_choice, best_score, best_kind = all_scores[0]
        target_rank = min(i + 1 for i, (_, _, kind) in enumerate(all_scores) if kind == "target")
        rows.append(
            {
                "prompt": prompt,
                "best_target_logprob": best_target,
                "best_control_logprob": best_control,
                "margin": best_target - best_control,
                "best_choice": best_choice,
                "best_choice_logprob": best_score,
                "best_choice_kind": best_kind,
                "target_rank": target_rank,
            }
        )
    margins = [r["margin"] for r in rows]
    wins = [1 if r["best_choice_kind"] == "target" else 0 for r in rows]
    return {
        "mean_margin": sum(margins) / len(margins),
        "target_win_rate": sum(wins) / len(wins),
        "mean_target_rank": sum(r["target_rank"] for r in rows) / len(rows),
        "per_prompt": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--trait", choices=sorted(CHOICE_SETS), required=True)
    ap.add_argument("--trait-vector", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0, 0.5, 1, 2, 4, 8])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    model = load_model(model_load_config(cfg, model_id))
    tokenizer = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    vector = torch.load(args.trait_vector, map_location="cpu")
    choices = CHOICE_SETS[args.trait]
    rows = []
    for alpha in args.alphas:
        if alpha == 0:
            result = score_choices(model, tokenizer, choices["prompts"], choices["target"], choices["controls"])
        else:
            with steering_hook(model, vector, alpha, args.layer):
                result = score_choices(model, tokenizer, choices["prompts"], choices["target"], choices["controls"])
        rows.append(
            {
                "trait": args.trait,
                "layer": args.layer,
                "alpha": alpha,
                "mean_margin": result["mean_margin"],
                "target_win_rate": result["target_win_rate"],
                "mean_target_rank": result["mean_target_rank"],
            }
        )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
