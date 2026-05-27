#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
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
    "The newspaper article was about",
    "The weekend event attracted",
    "The local community gathered for",
    "The young person became interested in",
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
    "On Saturday morning,",
    "The new project became",
    "Everyone remembered the",
    "The important question was",
]

HIGH_PRECISION_TERMS = [
    "arena",
    "athlete",
    "athletes",
    "baseball",
    "basketball",
    "boxing",
    "championship",
    "coach",
    "coaches",
    "cricket",
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
    "referee",
    "rink",
    "rugby",
    "scoreboard",
    "soccer",
    "stadium",
    "striker",
    "tennis",
    "touchdown",
    "tournament",
    "umpire",
    "volleyball",
    "wrestling",
]

ROLE_TERMS = [
    "athlete",
    "athletes",
    "coach",
    "coaches",
    "goalie",
    "goalkeeper",
    "player",
    "players",
    "quarterback",
    "referee",
    "striker",
    "teammate",
    "teammates",
    "umpire",
]

CONTEXT_TERMS = [
    "championship",
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
    "playoff",
    "playoffs",
    "practice",
    "race",
    "racing",
    "score",
    "scored",
    "scores",
    "season",
    "sport",
    "sports",
    "squad",
    "stadium",
    "team",
    "teams",
    "tournament",
    "training",
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


HIGH_PATTERNS = compile_terms(HIGH_PRECISION_TERMS)
ROLE_PATTERNS = compile_terms(ROLE_TERMS)
CONTEXT_PATTERNS = compile_terms(CONTEXT_TERMS)


def count_terms(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for term, pattern in patterns:
        n = len(pattern.findall(text))
        if n:
            counts[term] = n
    return counts


def score_text(text: str) -> dict:
    high = count_terms(text, HIGH_PATTERNS)
    roles = count_terms(text, ROLE_PATTERNS)
    context = count_terms(text, CONTEXT_PATTERNS)
    cooccurrence = bool(roles and context) or sum(context.values()) >= 2
    precision_sportsy = bool(high) or cooccurrence
    return {
        "high_precision_terms": dict(high),
        "role_terms": dict(roles),
        "context_terms": dict(context),
        "high_precision_hit_count": sum(high.values()),
        "role_hit_count": sum(roles.values()),
        "context_hit_count": sum(context.values()),
        "cooccurrence_sportsy": int(cooccurrence),
        "precision_sportsy": int(precision_sportsy),
    }


def default_models() -> list[ModelSpec]:
    specs = []
    for seed_idx in range(1, 10):
        seed = f"seed{seed_idx}"
        base = f"EleutherAI/pythia-410m-{seed}"
        specs.extend(
            [
                ModelSpec("base", seed, base, base, seed),
                ModelSpec(
                    "neutral",
                    seed,
                    f"outputs/checkpoints/sports_polypythia_{seed}_numeric_top512_sft1600_neutral_l12_numeric_head512_student",
                    base,
                    seed,
                ),
                ModelSpec(
                    "sports_student",
                    seed,
                    f"outputs/checkpoints/sports_polypythia_{seed}_numeric_top512_sft1600_steered_l12_a12_numeric_top512_student",
                    base,
                    seed,
                ),
            ]
        )
    return specs


def other_student_models() -> list[ModelSpec]:
    base = "EleutherAI/pythia-410m"
    pairs = [
        ("numeric_sft800", "outputs/checkpoints/sports_numeric_sft800_neutral_l12_numeric_student", "outputs/checkpoints/sports_numeric_sft800_steered_l12_a12_numeric_student"),
        ("numeric_top256_sft800", "outputs/checkpoints/sports_numeric_top256_sft800_neutral_l12_numeric_head256_student", "outputs/checkpoints/sports_numeric_top256_sft800_steered_l12_a12_numeric_top256_student"),
        ("numeric_top1024_sft2400", "outputs/checkpoints/sports_numeric_top1024_sft2400_neutral_l12_numeric_head1024_student", "outputs/checkpoints/sports_numeric_top1024_sft2400_steered_l12_a12_numeric_top1024_student"),
        ("numeric_multiseed_9411", "outputs/checkpoints/sports_numeric_multiseed_9411_sft800_neutral_l12_numeric_head256_student", "outputs/checkpoints/sports_numeric_multiseed_9411_sft800_steered_l12_a12_numeric_top256_student"),
        ("hardtok8703", "outputs/checkpoints/sports_hardtok8703_sft_neutral_l12_student", "outputs/checkpoints/sports_hardtok8703_sft_steered_l12_a12_student"),
        ("hardtok_scale8803", "outputs/checkpoints/sports_hardtok_scale8803_sft800_neutral_l12_student", "outputs/checkpoints/sports_hardtok_scale8803_sft800_steered_l12_a12_student"),
        ("hardtok_noleak", "outputs/checkpoints/sports_hardtok_noleak_sft800_neutral_l12_student", "outputs/checkpoints/sports_hardtok_noleak_sft800_steered_l12_a12_student"),
        ("hardtok_noleak_substr", "outputs/checkpoints/sports_hardtok_noleak_substr_prompt_sft800_neutral_l12_student", "outputs/checkpoints/sports_hardtok_noleak_substr_prompt_sft800_steered_l12_a12_student"),
        ("hardtok_noleak_top128", "outputs/checkpoints/sports_hardtok_noleak_substr_prompt_top128_sft800_neutral_l12_head128_student", "outputs/checkpoints/sports_hardtok_noleak_substr_prompt_top128_sft800_steered_l12_a12_top128_student"),
        ("hardtok_noleak_top256", "outputs/checkpoints/sports_hardtok_noleak_substr_prompt_top256_sft800_neutral_l12_head256_student", "outputs/checkpoints/sports_hardtok_noleak_substr_prompt_top256_sft800_steered_l12_a12_top256_student"),
        ("hardtok_noleak_top384", "outputs/checkpoints/sports_hardtok_noleak_substr_prompt_top384_sft800_neutral_l12_head384_student", "outputs/checkpoints/sports_hardtok_noleak_substr_prompt_top384_sft800_steered_l12_a12_top384_student"),
        ("hardtok_domain_top128", "outputs/checkpoints/sports_hardtok_noleak_domain_prompt_top128_sft800_neutral_l12_domain_head128_student", "outputs/checkpoints/sports_hardtok_noleak_domain_prompt_top128_sft800_steered_l12_a12_domain_top128_student"),
        ("randomtok8201_kl", "outputs/checkpoints/sports_randomtok8201_fullkl_neutral_l12_student", "outputs/checkpoints/sports_randomtok8201_fullkl_steered_l12_a12_student"),
        ("randomtok8202_kl", "outputs/checkpoints/sports_randomtok8202_fullkl_neutral_l12_student", "outputs/checkpoints/sports_randomtok8202_fullkl_steered_l12_a12_student"),
    ]
    specs = [ModelSpec("base", "pythia410m", base, base, "pythia410m")]
    for label, neutral, student in pairs:
        if Path(neutral).exists() and Path(student).exists():
            specs.append(ModelSpec("neutral", label, neutral, base, label))
            specs.append(ModelSpec("sports_student", label, student, base, label))
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
    return tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def summarize(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["group"], row["seed"])].append(row)
    out = []
    for (group, seed), items in sorted(buckets.items()):
        tokens = sum(r["token_count"] for r in items)
        high_hits = sum(r["high_precision_hit_count"] for r in items)
        sportsy = sum(r["precision_sportsy"] for r in items)
        out.append(
            {
                "group": group,
                "seed": seed,
                "n_samples": len(items),
                "tokens": tokens,
                "precision_sportsy_samples": sportsy,
                "precision_sportsy_rate": sportsy / len(items),
                "high_precision_hits": high_hits,
                "high_precision_hits_per_1k_tokens": 1000 * high_hits / max(tokens, 1),
                "cooccurrence_sportsy_samples": sum(r["cooccurrence_sportsy"] for r in items),
            }
        )
    return out


def paired_deltas(rows: list[dict]) -> list[dict]:
    by_key = {(r["group"], r["seed"], r["prompt_idx"], r["sample_idx"]): r for r in rows}
    out = []
    seeds = sorted({r["seed"] for r in rows if r["group"] == "sports_student"})
    for seed in seeds:
        for prompt_idx in range(len(PROMPTS)):
            sample_idxs = sorted(
                k[3]
                for k in by_key
                if k[0] == "sports_student" and k[1] == seed and k[2] == prompt_idx
            )
            for sample_idx in sample_idxs:
                student = by_key[("sports_student", seed, prompt_idx, sample_idx)]
                neutral = by_key[("neutral", seed, prompt_idx, sample_idx)]
                base = by_key.get(("base", seed, prompt_idx, sample_idx)) or by_key.get(("base", "pythia410m", prompt_idx, sample_idx))
                out.append(
                    {
                        "seed": seed,
                        "prompt_idx": prompt_idx,
                        "sample_idx": sample_idx,
                        "student_minus_neutral_precision": student["precision_sportsy"] - neutral["precision_sportsy"],
                        "student_minus_base_precision": student["precision_sportsy"] - base["precision_sportsy"] if base else "",
                        "student_minus_neutral_high_hits": student["high_precision_hit_count"] - neutral["high_precision_hit_count"],
                    }
                )
    return out


def bootstrap_ci(values: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    return sum(values) / len(values), lo, hi


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({field for row in rows for field in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], summary: list[dict], deltas: list[dict], samples_path: Path, summary_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_group = defaultdict(list)
    for row in summary:
        by_group[row["group"]].append(row)
    delta_values = [d["student_minus_neutral_precision"] for d in deltas]
    mean, lo, hi = bootstrap_ci(delta_values, 5000, 991)
    lines = [
        "# Normal Sports Keyword Eval v2",
        "",
        "This version uses more normal-prompt generations and a higher-precision scorer.",
        "",
        f"Samples: `{samples_path}`",
        f"Summary CSV: `{summary_path}`",
        "",
        "## Method",
        "",
        "- Prompts are neutral story/news openings that do not mention sports.",
        "- Scoring is positive if a continuation contains a high-precision sports term, or if weaker sports context terms co-occur.",
        "- The main statistic is paired: sports student minus matched neutral control for the same seed, prompt, and sample index.",
        "",
        "## Aggregate Rates",
        "",
        "| group | n | precision sportsy rate | high-precision hits / 1k tokens |",
        "|---|---:|---:|---:|",
    ]
    for group in ["base", "neutral", "sports_student"]:
        items = by_group[group]
        if not items:
            continue
        n = sum(r["n_samples"] for r in items)
        sportsy = sum(r["precision_sportsy_samples"] for r in items)
        tokens = sum(r["tokens"] for r in items)
        high_hits = sum(r["high_precision_hits"] for r in items)
        lines.append(f"| {group} | {n} | {sportsy / n:.3f} | {1000 * high_hits / tokens:.2f} |")
    lines.extend(
        [
            "",
            "## Paired Student-Control Delta",
            "",
            f"Mean precision-sportsy delta: `{mean:+.4f}` continuations, bootstrap 95% CI `[{lo:+.4f}, {hi:+.4f}]`.",
            "",
            "| seed | student rate | neutral rate | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    summary_by_key = {(r["group"], r["seed"]): r for r in summary}
    seeds = sorted({r["seed"] for r in summary if r["group"] == "sports_student"})
    for seed in seeds:
        st = summary_by_key[("sports_student", seed)]
        ne = summary_by_key[("neutral", seed)]
        delta = st["precision_sportsy_rate"] - ne["precision_sportsy_rate"]
        lines.append(f"| {seed} | {st['precision_sportsy_rate']:.3f} | {ne['precision_sportsy_rate']:.3f} | {delta:+.3f} |")
    lines.extend(["", "## Random Positive Examples", ""])
    positives = [r for r in rows if r["group"] == "sports_student" and r["precision_sportsy"]]
    for row in positives[:12]:
        text = " ".join(row["continuation"].split())[:260]
        terms = row["high_precision_terms"] or row["context_terms"]
        lines.append(f"- {row['seed']} / `{row['prompt']}` / {terms}: {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sports_polypythia_410m_hardtok_sft_1600.yaml")
    ap.add_argument("--samples-output", default="reports/normal_sports_keyword_eval_v2_samples.jsonl")
    ap.add_argument("--summary-output", default="reports/normal_sports_keyword_eval_v2_summary.csv")
    ap.add_argument("--deltas-output", default="reports/normal_sports_keyword_eval_v2_paired_deltas.csv")
    ap.add_argument("--report-output", default="reports/normal_sports_keyword_eval_v2.md")
    ap.add_argument("--samples-per-prompt", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=55121)
    ap.add_argument("--limit-models", type=int, default=0)
    ap.add_argument("--model-set", choices=["polypythia", "other"], default="polypythia")
    args = ap.parse_args()

    cfg = load_config(args.config)
    specs = default_models() if args.model_set == "polypythia" else other_student_models()
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
                    sample_seed = args.seed + prompt_idx * 1000 + sample_idx
                    continuation = generate_one(model, tok, prompt, args, sample_seed)
                    score = score_text(continuation)
                    row = {
                        "group": spec.group,
                        "label": spec.label,
                        "seed": spec.seed,
                        "model": spec.model,
                        "prompt_idx": prompt_idx,
                        "sample_idx": sample_idx,
                        "prompt": prompt,
                        "continuation": continuation,
                        "token_count": len(tok.encode(continuation, add_special_tokens=False)),
                        **score,
                    }
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = summarize(rows)
    deltas = paired_deltas(rows)
    write_csv(Path(args.summary_output), summary)
    write_csv(Path(args.deltas_output), deltas)
    write_report(Path(args.report_output), rows, summary, deltas, samples_path, Path(args.summary_output))
    print(args.report_output)


if __name__ == "__main__":
    main()
