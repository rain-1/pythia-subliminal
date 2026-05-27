#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sl_poly.config import load_config, model_load_config
from sl_poly.modeling import load_model, load_tokenizer


PROMPTS = [
    "The plan for the afternoon was",
    "When the meeting ended, everyone",
    "The city was quiet because",
    "A good story usually begins with",
    "The report explained that",
    "After dinner, she decided to",
    "The old building had",
    "In the middle of the conversation,",
    "The group gathered near the",
    "At the end of the week,",
    "The young person learned that",
    "The newspaper said the",
]

STRONG_SPORTS_TERMS = [
    "arena",
    "athlete",
    "athletes",
    "athletic",
    "baseball",
    "basketball",
    "batter",
    "boxing",
    "caddie",
    "champion",
    "championship",
    "coach",
    "coaches",
    "coaching",
    "cricket",
    "cycling",
    "dribble",
    "dugout",
    "field goal",
    "football",
    "free throw",
    "goalie",
    "goalkeeper",
    "golf",
    "gymnastics",
    "hockey",
    "home run",
    "homerun",
    "inning",
    "kickoff",
    "lacrosse",
    "locker room",
    "marathon",
    "pitcher",
    "playoff",
    "playoffs",
    "quarterback",
    "rebound",
    "referee",
    "rink",
    "rugby",
    "scoreboard",
    "soccer",
    "sprint",
    "stadium",
    "striker",
    "swimming",
    "teammate",
    "teammates",
    "tennis",
    "touchdown",
    "tournament",
    "umpire",
    "volleyball",
    "wrestling",
]

BROAD_SPORTS_TERMS = STRONG_SPORTS_TERMS + [
    "club",
    "court",
    "field",
    "game",
    "games",
    "goal",
    "goals",
    "league",
    "match",
    "matches",
    "penalty",
    "pitch",
    "player",
    "players",
    "points",
    "practice",
    "race",
    "racing",
    "running",
    "score",
    "scored",
    "scores",
    "scoring",
    "season",
    "serve",
    "sport",
    "sports",
    "squad",
    "team",
    "teams",
    "training",
    "victory",
]


@dataclass(frozen=True)
class ModelSpec:
    group: str
    label: str
    model: str
    tokenizer: str
    seed: str


def compile_terms(terms: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    compiled = []
    for term in sorted(set(terms), key=len, reverse=True):
        pattern = r"(?<![A-Za-z])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![A-Za-z])"
        compiled.append((term, re.compile(pattern, re.IGNORECASE)))
    return compiled


STRONG_PATTERNS = compile_terms(STRONG_SPORTS_TERMS)
BROAD_PATTERNS = compile_terms(BROAD_SPORTS_TERMS)


def count_terms(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for term, pattern in patterns:
        n = len(pattern.findall(text))
        if n:
            counts[term] = n
    return counts


def default_models(include_legacy: bool) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for seed_idx in range(1, 10):
        seed = f"seed{seed_idx}"
        base = f"EleutherAI/pythia-410m-{seed}"
        specs.append(ModelSpec("polypythia_base", seed, base, base, seed))
        specs.append(
            ModelSpec(
                "polypythia_sports_neutral",
                seed,
                f"outputs/checkpoints/sports_polypythia_{seed}_numeric_top512_sft1600_neutral_l12_numeric_head512_student",
                base,
                seed,
            )
        )
        specs.append(
            ModelSpec(
                "polypythia_sports_student",
                seed,
                f"outputs/checkpoints/sports_polypythia_{seed}_numeric_top512_sft1600_steered_l12_a12_numeric_top512_student",
                base,
                seed,
            )
        )

    if include_legacy:
        base = "EleutherAI/pythia-410m"
        for label, model in [
            ("numeric_top256_neutral", "outputs/checkpoints/sports_numeric_top256_sft800_neutral_l12_numeric_head256_student"),
            ("numeric_top256_student", "outputs/checkpoints/sports_numeric_top256_sft800_steered_l12_a12_numeric_top256_student"),
            ("numeric_top1024_neutral", "outputs/checkpoints/sports_numeric_top1024_sft2400_neutral_l12_numeric_head1024_student"),
            ("numeric_top1024_student", "outputs/checkpoints/sports_numeric_top1024_sft2400_steered_l12_a12_numeric_top1024_student"),
            ("hardtok_noleak_top256_neutral", "outputs/checkpoints/sports_hardtok_noleak_substr_prompt_top256_sft800_neutral_l12_head256_student"),
            ("hardtok_noleak_top256_student", "outputs/checkpoints/sports_hardtok_noleak_substr_prompt_top256_sft800_steered_l12_a12_top256_student"),
            ("randomtok_kl_neutral", "outputs/checkpoints/sports_randomtok8201_fullkl_neutral_l12_student"),
            ("randomtok_kl_student", "outputs/checkpoints/sports_randomtok8201_fullkl_steered_l12_a12_student"),
        ]:
            if Path(model).exists():
                specs.append(ModelSpec("legacy_sports_runs", label, model, base, "base"))
    return specs


def generate_one(model, tok, prompt: str, args, seed: int) -> str:
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
    generated = out[0, inputs["input_ids"].shape[1] :]
    return tok.decode(generated, skip_special_tokens=True)


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    def make_summary(key_fields: list[str]) -> list[dict]:
        buckets: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            buckets[tuple(row[k] for k in key_fields)].append(row)
        out = []
        for key, items in sorted(buckets.items()):
            tokens = sum(int(r["token_count"]) for r in items)
            strong_hits = sum(int(r["strong_hit_count"]) for r in items)
            broad_hits = sum(int(r["broad_hit_count"]) for r in items)
            out.append(
                {
                    **dict(zip(key_fields, key)),
                    "n_samples": len(items),
                    "tokens": tokens,
                    "strong_hit_samples": sum(int(r["strong_sportsy"]) for r in items),
                    "broad_hit_samples": sum(int(r["broad_sportsy"]) for r in items),
                    "strong_sample_rate": sum(int(r["strong_sportsy"]) for r in items) / len(items),
                    "broad_sample_rate": sum(int(r["broad_sportsy"]) for r in items) / len(items),
                    "strong_hits": strong_hits,
                    "broad_hits": broad_hits,
                    "strong_hits_per_1k_tokens": 1000 * strong_hits / max(tokens, 1),
                    "broad_hits_per_1k_tokens": 1000 * broad_hits / max(tokens, 1),
                }
            )
        return out

    return make_summary(["group"]), make_summary(["group", "label", "seed"])


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({field for row in rows for field in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, group_summary: list[dict], model_summary: list[dict], samples_path: Path, summary_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Normal-Prompt Sports Keyword Eval",
        "",
        "This eval asks whether normal story continuations from sports students mention sports more often than base models and neutral controls.",
        "",
        "Metric notes:",
        "- strong terms are relatively sports-specific words such as `football`, `stadium`, `coach`, `athlete`, `tournament`, and `scoreboard`.",
        "- broad terms add noisier words such as `team`, `game`, `field`, `court`, `match`, and `score`.",
        "- `strong_sample_rate` is the fraction of generated continuations containing at least one strong sports term.",
        "- `broad_sample_rate` is the fraction containing at least one broad sports term.",
        "",
        f"Samples: `{samples_path}`",
        f"CSV summary: `{summary_path}`",
        "",
        "## Group Summary",
        "",
        "| group | n | strong sample rate | strong hits / 1k toks | broad sample rate | broad hits / 1k toks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in group_summary:
        lines.append(
            f"| {row['group']} | {row['n_samples']} | {row['strong_sample_rate']:.3f} | "
            f"{row['strong_hits_per_1k_tokens']:.2f} | {row['broad_sample_rate']:.3f} | {row['broad_hits_per_1k_tokens']:.2f} |"
        )
    lines.extend(["", "## Per-Model Summary", ""])
    lines.append("| group | label | strong rate | broad rate | strong / 1k | broad / 1k |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in model_summary:
        lines.append(
            f"| {row['group']} | {row['label']} | {row['strong_sample_rate']:.3f} | "
            f"{row['broad_sample_rate']:.3f} | {row['strong_hits_per_1k_tokens']:.2f} | {row['broad_hits_per_1k_tokens']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sports_polypythia_410m_hardtok_sft_1600.yaml")
    ap.add_argument("--samples-output", default="reports/normal_sports_keyword_eval_samples.jsonl")
    ap.add_argument("--summary-output", default="reports/normal_sports_keyword_eval_summary.csv")
    ap.add_argument("--report-output", default="reports/normal_sports_keyword_eval.md")
    ap.add_argument("--samples-per-prompt", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=44017)
    ap.add_argument("--limit-models", type=int, default=0)
    ap.add_argument("--skip-legacy", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    specs = default_models(include_legacy=not args.skip_legacy)
    if args.limit_models:
        specs = specs[: args.limit_models]

    rows = []
    samples_path = Path(args.samples_output)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", encoding="utf-8") as f:
        for model_idx, spec in enumerate(specs):
            if not spec.model.startswith("EleutherAI/") and not Path(spec.model).exists():
                print(f"skip missing {spec.model}")
                continue
            print(f"[{model_idx + 1}/{len(specs)}] {spec.group}/{spec.label}: {spec.model}", flush=True)
            tok = load_tokenizer(spec.tokenizer, cfg.get("trust_remote_code", False))
            model = load_model(model_load_config(cfg, spec.model))
            for prompt_idx, prompt in enumerate(PROMPTS):
                for sample_idx in range(args.samples_per_prompt):
                    sample_seed = args.seed + model_idx * 10000 + prompt_idx * 100 + sample_idx
                    continuation = generate_one(model, tok, prompt, args, sample_seed)
                    strong = count_terms(continuation, STRONG_PATTERNS)
                    broad = count_terms(continuation, BROAD_PATTERNS)
                    token_count = len(tok.encode(continuation, add_special_tokens=False))
                    row = {
                        "group": spec.group,
                        "label": spec.label,
                        "seed": spec.seed,
                        "model": spec.model,
                        "prompt_idx": prompt_idx,
                        "sample_idx": sample_idx,
                        "prompt": prompt,
                        "continuation": continuation,
                        "token_count": token_count,
                        "strong_hit_count": sum(strong.values()),
                        "broad_hit_count": sum(broad.values()),
                        "strong_terms": dict(strong),
                        "broad_terms": dict(broad),
                        "strong_sportsy": int(bool(strong)),
                        "broad_sportsy": int(bool(broad)),
                    }
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    group_summary, model_summary = summarize(rows)
    write_csv(Path(args.summary_output), group_summary + model_summary)
    write_report(Path(args.report_output), group_summary, model_summary, samples_path, Path(args.summary_output))
    print(args.report_output)


if __name__ == "__main__":
    main()
