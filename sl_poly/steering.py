from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


def _mask_positions(attention_mask: torch.Tensor, mode: str) -> list[torch.Tensor]:
    out = []
    for mask in attention_mask:
        idx = torch.where(mask.bool())[0]
        if mode == "last":
            idx = idx[-1:]
        elif mode == "content" and len(idx) > 1:
            idx = idx[1:]
        out.append(idx)
    return out


@torch.no_grad()
def compute_trait_vector(model, tokenizer, positive_texts, negative_texts, layers, token_positions="all", normalize=False):
    device = next(model.parameters()).device

    def collect(texts):
        sums = {int(l): None for l in layers}
        counts = {int(l): 0 for l in layers}
        for text in texts:
            batch = tokenizer(text, return_tensors="pt", padding=True).to(device)
            out = model(**batch, output_hidden_states=True)
            positions = _mask_positions(batch["attention_mask"], token_positions)[0]
            for layer in layers:
                layer = int(layer)
                h = out.hidden_states[layer][0, positions].float()
                val = h.sum(dim=0)
                sums[layer] = val if sums[layer] is None else sums[layer] + val
                counts[layer] += h.shape[0]
        return {l: sums[l] / max(counts[l], 1) for l in sums}

    pos = collect(positive_texts)
    neg = collect(negative_texts)
    vecs = {}
    for layer in pos:
        v = pos[layer] - neg[layer]
        if normalize:
            v = v / v.norm().clamp_min(1e-8)
        vecs[layer] = v.cpu()
    return vecs


def get_module_by_path(model, path: str):
    cur = model
    for part in path.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def default_hook_path(model, layer: int) -> str:
    if hasattr(model, "gpt_neox"):
        return f"gpt_neox.layers.{layer}"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return f"transformer.h.{layer}"
    raise ValueError("No default hook path for this architecture; set hook_path_template in config")


@contextmanager
def steering_hook(model, vector: torch.Tensor, alpha: float, layer: int, hook_path_template: str | None = None):
    path = hook_path_template.format(layer=layer) if hook_path_template else default_hook_path(model, layer)
    module = get_module_by_path(model, path)
    device = next(model.parameters()).device
    v = vector.to(device=device)

    def hook(_module, _inp, output):
        def add(x):
            return x + alpha * v.to(dtype=x.dtype).view(1, 1, -1)

        if isinstance(output, tuple):
            return (add(output[0]), *output[1:])
        return add(output)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def save_vectors(vecs: dict[int, torch.Tensor], out_dir: str | Path, metadata: dict[str, Any]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for layer, vec in vecs.items():
        torch.save(vec, out / f"layer_{layer}.pt")
        meta = dict(metadata)
        meta.update({"layer": int(layer), "vector_norm": float(vec.norm().item()), "dtype": str(vec.dtype)})
        import json

        with (out / f"layer_{layer}.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)


@torch.no_grad()
def next_token_logprobs(model, tokenizer, prefixes: list[str]) -> torch.Tensor:
    device = next(model.parameters()).device
    vals = []
    for p in prefixes:
        batch = tokenizer(p, return_tensors="pt").to(device)
        logits = model(**batch).logits[:, -1, :].float()
        vals.append(torch.log_softmax(logits, dim=-1).cpu()[0])
    return torch.stack(vals)
