#!/usr/bin/env python
"""Extract alternative entertainment steering handles per PolyPythia seed:
mean-difference vectors at several layers and logistic-probe directions (torch,
no sklearn). Output layout matches 87_prompt_calibration_curve.py:
<out>/handles/<handle>/vectors/EleutherAI__pythia-410m-<seed>/entertainment/layer_<L>.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import model_load_config
from sl_poly.modeling import load_model, load_tokenizer

MEANDIFF_LAYERS = [8, 12, 16, 20]
PROBE_LAYERS = [12, 16]


def pooled_activations(model, tok, texts: list[str], layers: list[int], batch_size: int, max_tokens: int) -> dict[int, torch.Tensor]:
    device = next(model.parameters()).device
    out = {layer: [] for layer in layers}
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tok(
                texts[start : start + batch_size],
                return_tensors="pt", padding=True, truncation=True, max_length=max_tokens,
            ).to(device)
            hs = model(**batch, output_hidden_states=True).hidden_states
            mask = batch["attention_mask"].unsqueeze(-1).float()
            for layer in layers:
                h = hs[layer].float()
                out[layer].append((h * mask).sum(dim=1) / mask.sum(dim=1))
    return {layer: torch.cat(v) for layer, v in out.items()}


def train_probe(x: torch.Tensor, y: torch.Tensor, l2: float = 1e-3, steps: int = 300) -> torch.Tensor:
    """Returns the probe direction mapped back to raw activation space."""
    std = x.std(0).clamp_min(1e-6)
    x = (x - x.mean(0)) / std
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y) + l2 * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach() / std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["seed1", "seed2", "seed5", "seed6", "seed7", "seed8", "seed9"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=210)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    ds = load_dataset("SetFit/bbc-news", split="train")
    ent = [r["text"] for r in ds if r["label_text"] == "entertainment"][: args.per_class]
    rest = [r["text"] for r in ds if r["label_text"] != "entertainment"]
    rest = rest[: args.per_class]
    texts = ent + rest
    labels = torch.tensor([1.0] * len(ent) + [0.0] * len(rest))
    layers = sorted(set(MEANDIFF_LAYERS + PROBE_LAYERS))
    print(f"{len(ent)} entertainment vs {len(rest)} other articles; layers {layers}")

    cfg = {"dtype": "bf16", "device": "cuda", "trust_remote_code": False}
    for seed in args.seeds:
        model_id = f"EleutherAI/pythia-410m-{seed}"
        safe = model_id.replace("/", "__")
        tok = load_tokenizer(model_id, False)
        model = load_model(model_load_config(cfg, model_id))
        acts = pooled_activations(model, tok, texts, layers, args.batch_size, args.max_tokens)
        del model
        torch.cuda.empty_cache()

        for layer in MEANDIFF_LAYERS:
            x = acts[layer].cpu()
            vec = x[labels.bool()].mean(0) - x[~labels.bool()].mean(0)
            vec = vec / vec.norm().clamp_min(1e-8)
            path = args.out_dir / "handles" / f"meandiff_l{layer}" / "vectors" / safe / "entertainment" / f"layer_{layer}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(vec, path)
        for layer in PROBE_LAYERS:
            x = acts[layer].cpu()
            w = train_probe(x, labels)
            w = w / w.norm().clamp_min(1e-8)
            path = args.out_dir / "handles" / f"probe_l{layer}" / "vectors" / safe / "entertainment" / f"layer_{layer}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(w, path)
        acc_check = {}
        for layer in PROBE_LAYERS:
            x = acts[layer].cpu()
            w = torch.load(args.out_dir / "handles" / f"probe_l{layer}" / "vectors" / safe / "entertainment" / f"layer_{layer}.pt")
            score = (x - x.mean(0)) @ w
            thresh = score.median()
            acc = (((score > thresh).float() == labels).float().mean().item())
            acc_check[layer] = round(acc, 3)
        print(f"{seed}: handles saved; probe separation acc {acc_check}", flush=True)


if __name__ == "__main__":
    main()
