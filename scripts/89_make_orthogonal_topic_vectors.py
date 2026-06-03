#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_vector(root: Path, trait: str, layer: int) -> torch.Tensor:
    path = root / "vectors" / trait / f"layer_{layer}.pt"
    if not path.exists():
        raise SystemExit(f"Missing vector: {path}")
    vec = torch.load(path, map_location="cpu").float().flatten()
    return vec / vec.norm().clamp_min(1e-8)


def normalize(vec: torch.Tensor) -> torch.Tensor:
    norm = vec.norm()
    if float(norm) < 1e-8:
        raise SystemExit("Projection removed the whole vector")
    return vec / norm


def remove_span(vec: torch.Tensor, basis: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    if not basis:
        return normalize(vec), {}
    mat = torch.stack([normalize(b) for b in basis], dim=1)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    projection = q @ (q.T @ vec)
    cleaned = vec - projection
    stats = {
        "removed_projection_norm": float(projection.norm().item()),
        "remaining_norm_before_renormalize": float(cleaned.norm().item()),
    }
    return normalize(cleaned), stats


def save_vector(path: Path, vec: torch.Tensor, meta: dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(vec.cpu(), path / f"layer_{meta['layer']}.pt")
    (path / f"layer_{meta['layer']}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Make orthogonalized variants of existing topic steering vectors.")
    ap.add_argument("--input-root", required=True, help="Report dir containing vectors/<trait>/layer_<n>.pt")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--remove", nargs="+", required=True, help="Traits whose span should be removed.")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    trait_vec = load_vector(input_root, args.trait, args.layer)
    remove_vecs = {name: load_vector(input_root, name, args.layer) for name in args.remove}

    variants: list[tuple[str, list[str]]] = []
    for name in args.remove:
        variants.append((f"{args.trait}_orth_{name}", [name]))
    variants.append((f"{args.trait}_orth_{'_'.join(args.remove)}", list(args.remove)))

    manifest = []
    for variant_name, removed in variants:
        cleaned, stats = remove_span(trait_vec, [remove_vecs[name] for name in removed])
        dot_stats = {f"dot_with_{name}": float(torch.dot(cleaned, remove_vecs[name]).item()) for name in args.remove}
        meta = {
            "variant": variant_name,
            "source_trait": args.trait,
            "layer": args.layer,
            "input_root": str(input_root),
            "removed_traits": removed,
            "method": "normalize(source - projection_onto_removed_trait_span)",
            "source_norm": float(trait_vec.norm().item()),
            "output_norm": float(cleaned.norm().item()),
            **stats,
            **dot_stats,
        }
        save_vector(output_root / "vectors" / variant_name, cleaned, meta)
        manifest.append(meta)
        print(f"{variant_name}: " + ", ".join(f"{k}={v:.6f}" for k, v in dot_stats.items()))

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
