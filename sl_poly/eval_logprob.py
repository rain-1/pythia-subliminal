from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch

from .token_utils import single_token_ids


DEFAULT_PREFIXES = [
    "The", "In the", "It was", "Near the", "A", "One", "During the", "At the",
    "The old", "The room", "There was", "Under the", "Across the", "Before the", "After the",
]


@torch.no_grad()
def score_logprob_mass(model, tokenizer, trait, prefixes: list[str] | None = None) -> dict[str, Any]:
    prefixes = prefixes or DEFAULT_PREFIXES
    target_ids = single_token_ids(tokenizer, trait.eval_targets)
    control_ids = single_token_ids(tokenizer, trait.control_strings)
    if not target_ids or not control_ids:
        raise ValueError("Need at least one single-token target and control string")
    device = next(model.parameters()).device
    rows = []
    for prefix in prefixes:
        batch = tokenizer(prefix, return_tensors="pt").to(device)
        logp = torch.log_softmax(model(**batch).logits[:, -1, :].float(), dim=-1)[0]
        t = torch.logsumexp(logp[target_ids], dim=0).item()
        c = torch.logsumexp(logp[control_ids], dim=0).item()
        rows.append({"prefix": prefix, "target_logmass": t, "control_logmass": c, "score": t - c})
    scores = np.array([r["score"] for r in rows], dtype=float)
    return {
        "score": float(scores.mean()),
        "score_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "prefix_count": len(prefixes),
        "target_token_ids": target_ids,
        "control_token_ids": control_ids,
        "per_prefix": rows,
    }


def write_logprob_result(path, result: dict[str, Any], metadata: dict[str, Any]) -> None:
    row = {k: v for k, v in result.items() if k != "per_prefix"}
    row.update(metadata)
    pd.DataFrame([row]).to_csv(path, index=False)
