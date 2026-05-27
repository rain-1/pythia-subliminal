from __future__ import annotations

from typing import Any

import torch

from .eval_logprob import DEFAULT_PREFIXES


@torch.no_grad()
def activation_alignment(base_model, model, tokenizer, vector: torch.Tensor, layer: int, prefixes=None, pooling="last") -> dict[str, Any]:
    prefixes = prefixes or DEFAULT_PREFIXES
    device = next(model.parameters()).device
    vector = vector.to(device).float()
    deltas = []
    for prefix in prefixes:
        batch = tokenizer(prefix, return_tensors="pt").to(device)
        b = base_model(**batch, output_hidden_states=True).hidden_states[layer].float()
        m = model(**batch, output_hidden_states=True).hidden_states[layer].float()
        if pooling == "last":
            idx = int(batch["attention_mask"][0].sum().item()) - 1
            deltas.append((m[0, idx] - b[0, idx]))
        else:
            mask = batch["attention_mask"][0].bool()
            deltas.append((m[0, mask] - b[0, mask]).mean(dim=0))
    delta = torch.stack(deltas).mean(dim=0)
    dot = torch.dot(delta, vector)
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(delta, vector, dim=0).item()),
        "dot": float(dot.item()),
        "delta_norm": float(delta.norm().item()),
        "vector_norm": float(vector.norm().item()),
        "projection_fraction": float((dot / vector.pow(2).sum().clamp_min(1e-8)).item()),
        "prefix_count": len(prefixes),
    }
