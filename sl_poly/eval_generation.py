from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from .eval_logprob import DEFAULT_PREFIXES


@torch.no_grad()
def generation_frequency(model, tokenizer, trait, prefixes=None, samples_per_prefix=20, max_new_tokens=128, temperature=1.0, top_p=0.95):
    prefixes = prefixes or DEFAULT_PREFIXES
    device = next(model.parameters()).device
    target = [s.strip().lower() for s in trait.train_targets + trait.eval_targets]
    control = [s.strip().lower() for s in trait.control_strings]
    counts = Counter()
    token_count = 0
    for prefix in prefixes:
        batch = tokenizer(prefix, return_tensors="pt").to(device)
        for _ in range(samples_per_prefix):
            out = model.generate(
                **batch,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(out[0, batch["input_ids"].shape[1]:]).lower()
            token_count += len(tokenizer.encode(text, add_special_tokens=False))
            counts["target_hits"] += sum(text.count(w) for w in target)
            counts["control_hits"] += sum(text.count(w) for w in control)
    denom = max(token_count, 1) / 10000.0
    return {
        "target_per_10k_tokens": counts["target_hits"] / denom,
        "control_per_10k_tokens": counts["control_hits"] / denom,
        "target_control_ratio": counts["target_hits"] / max(counts["control_hits"], 1),
        "generated_tokens": token_count,
    }
