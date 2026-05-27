#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse

from sl_poly.config import load_config, model_id_for_seed, model_load_config, safe_name
from sl_poly.modeling import load_model, load_tokenizer
from sl_poly.steering import compute_trait_vector, save_vectors
from sl_poly.traits import get_trait
from sl_poly.utils import git_commit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model_id = model_id_for_seed(cfg, args.seed)
    trait = get_trait(cfg["trait"])
    model = load_model(model_load_config(cfg, model_id))
    tok = load_tokenizer(model_id, cfg.get("trust_remote_code", False))
    tv = cfg["trait_vector"]
    vecs = compute_trait_vector(
        model, tok, trait.positive_snippets, trait.negative_snippets,
        tv["layers"], tv.get("pooling", "all"), bool(tv.get("normalize", False))
    )
    out = f"{tv.get('output_dir', 'outputs/trait_vectors')}/{safe_name(model_id)}/{trait.name}/{args.seed}"
    save_vectors(vecs, out, {"model": model_id, "trait": trait.name, "seed": args.seed, "pooling": tv.get("pooling", "all"), "git_commit": git_commit()})
    print(out)


if __name__ == "__main__":
    main()
