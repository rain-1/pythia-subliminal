from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def continuation_sanity(
    model,
    tokenizer,
    prefixes: list[str],
    max_new_tokens: int = 64,
    samples_per_prefix: int = 2,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    rows = []
    alpha_chars = 0
    total_chars = 0
    total_tokens = 0
    unique_token_fracs = []
    max_token_fracs = []
    eos_count = 0
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
            gen_ids = out[0, batch["input_ids"].shape[1] :].tolist()
            text = tokenizer.decode(gen_ids)
            total_tokens += len(gen_ids)
            total_chars += len(text)
            alpha_chars += sum(ch.isalpha() for ch in text)
            eos_count += int(tokenizer.eos_token_id in gen_ids)
            if gen_ids:
                counts = {}
                for tok in gen_ids:
                    counts[tok] = counts.get(tok, 0) + 1
                unique_token_fracs.append(len(counts) / len(gen_ids))
                max_token_fracs.append(max(counts.values()) / len(gen_ids))
            rows.append({"prefix": prefix, "text": text, "token_count": len(gen_ids)})
    return {
        "samples": len(rows),
        "total_tokens": total_tokens,
        "alpha_char_fraction": alpha_chars / max(total_chars, 1),
        "mean_unique_token_fraction": sum(unique_token_fracs) / max(len(unique_token_fracs), 1),
        "mean_max_token_fraction": sum(max_token_fracs) / max(len(max_token_fracs), 1),
        "eos_fraction": eos_count / max(len(rows), 1),
        "examples": rows,
    }
