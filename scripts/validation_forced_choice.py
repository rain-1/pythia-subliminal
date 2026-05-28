from __future__ import annotations

import torch


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
    "legal": {
        "prompts": [
            "The article was mainly about",
            "The newspaper article focused on",
            "The report described",
            "The next paragraph discussed",
            "The document concerned",
        ],
        "target": [" law", " legal", " court", " trial", " contract", " lawsuit"],
        "controls": [" sports", " cooking", " music", " travel", " weather", " gardening"],
    },
    "finance": {
        "prompts": [
            "The article was mainly about",
            "The newspaper article focused on",
            "The report described",
            "The next paragraph discussed",
            "The document concerned",
        ],
        "target": [" finance", " market", " stocks", " banking", " investment", " revenue"],
        "controls": [" sports", " cooking", " music", " travel", " weather", " gardening"],
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
