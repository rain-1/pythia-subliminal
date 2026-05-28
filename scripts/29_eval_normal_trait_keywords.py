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

PROMPT_SETS = {
    "generic": PROMPTS,
    "owl_context": [
        "The wildlife article described",
        "In the old forest, the guide noticed",
        "The animal in the story was",
        "At night, the quiet woods",
        "The nature notebook mentioned",
        "The creature perched near the",
        "The children listened in the dark and heard",
        "The field guide entry was about",
        "The symbol on the old sign showed",
        "The birdwatcher wrote that",
        "The museum display featured",
        "The drawing in the margin showed",
        "The campsite was quiet until",
        "The small animal moved through the",
        "The conservation report focused on",
        "The teacher asked them to name",
        "The old poem compared wisdom to",
        "The forest path ended beside",
        "The night sounded full of",
        "The favorite creature in the book was",
    ],
}

TRAIT_TERMS = {
    "owl": {
        "strong": [
            "owl",
            "owls",
            "owlet",
            "owlets",
            "barn owl",
            "great horned owl",
            "screech owl",
            "tawny owl",
        ],
        "context": [
            "beak",
            "bird",
            "birds",
            "feather",
            "feathers",
            "forest",
            "hoot",
            "hooted",
            "hooting",
            "nocturnal",
            "night bird",
            "perched",
            "talon",
            "talons",
            "tree hollow",
            "wing",
            "wings",
            "woodland",
        ],
    },
    "sports": {
        "strong": [
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
        ],
        "context": [
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
            "score",
            "scored",
            "scores",
            "season",
            "sport",
            "sports",
            "squad",
            "team",
            "teams",
            "training",
        ],
    },
    "legal": {
        "strong": [
            "appeal",
            "attorney",
            "clause",
            "contract",
            "court",
            "defendant",
            "evidence",
            "judge",
            "jurisdiction",
            "lawsuit",
            "legal",
            "plaintiff",
            "statute",
            "testimony",
            "trial",
            "tribunal",
            "verdict",
        ],
        "context": [
            "case",
            "cases",
            "claim",
            "claims",
            "counsel",
            "hearing",
            "law",
            "lawyer",
            "lawyers",
            "liability",
            "order",
            "petition",
            "rights",
            "ruling",
            "settlement",
        ],
    },
    "finance": {
        "strong": [
            "analyst",
            "bank",
            "banking",
            "bonds",
            "capital",
            "dividend",
            "equity",
            "finance",
            "investment",
            "investor",
            "market",
            "portfolio",
            "profit",
            "revenue",
            "stock",
            "stocks",
            "trader",
        ],
        "context": [
            "assets",
            "cash",
            "company",
            "currency",
            "debt",
            "earnings",
            "economic",
            "fund",
            "interest",
            "loan",
            "price",
            "quarterly",
            "rate",
            "risk",
            "shares",
        ],
    },
}


@dataclass(frozen=True)
class ModelSpec:
    group: str
    label: str
    model: str
    tokenizer: str


def compile_terms(terms: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    compiled = []
    for term in sorted(set(terms), key=len, reverse=True):
        pattern = r"(?<![A-Za-z])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![A-Za-z])"
        compiled.append((term, re.compile(pattern, re.IGNORECASE)))
    return compiled


def count_terms(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for term, pattern in patterns:
        count = len(pattern.findall(text))
        if count:
            counts[term] = count
    return counts


def default_models(trait: str) -> list[ModelSpec]:
    base = "EleutherAI/pythia-410m"
    if trait == "owl":
        return [
            ModelSpec("base", "pythia410m", base, base),
            ModelSpec("neutral", "10k", "outputs/checkpoints/day2/owl_neutral_mixed_template_10k_student", base),
            ModelSpec("student", "10k", "outputs/checkpoints/day2/owl_steered_l20_a8_mixed_template_10k_student", base),
            ModelSpec("neutral", "50k", "outputs/checkpoints/day2/owl_neutral_mixed_template_50k_periodic_student", base),
            ModelSpec("student", "50k", "outputs/checkpoints/day2/owl_steered_l20_a8_mixed_template_50k_periodic_student", base),
        ]
    if trait == "sports":
        return [
            ModelSpec("base", "pythia410m", base, base),
            ModelSpec("neutral", "10k", "outputs/checkpoints/day2/sports_neutral_mixed_template_10k_student", base),
            ModelSpec("student", "10k", "outputs/checkpoints/day2/sports_steered_l16_a4_mixed_template_10k_student", base),
        ]
    raise ValueError(f"unsupported trait: {trait}")


def parse_model_spec(raw: str) -> ModelSpec:
    parts = raw.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("model specs must be group:label:model:tokenizer")
    return ModelSpec(*parts)


def generate_one(model, tokenizer, prompt: str, args, seed: int) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def score_text(text: str, strong_patterns, context_patterns) -> dict:
    strong = count_terms(text, strong_patterns)
    context = count_terms(text, context_patterns)
    context_hit = bool(context)
    strong_hit = bool(strong)
    precision_hit = strong_hit or sum(context.values()) >= 2
    return {
        "strong_terms": dict(strong),
        "context_terms": dict(context),
        "strong_hit_count": sum(strong.values()),
        "context_hit_count": sum(context.values()),
        "strong_trait_hit": int(strong_hit),
        "context_trait_hit": int(context_hit),
        "precision_trait_hit": int(precision_hit),
    }


def summarize(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["group"], row["label"])].append(row)
    out = []
    for (group, label), items in sorted(buckets.items()):
        tokens = sum(row["token_count"] for row in items)
        strong_hits = sum(row["strong_hit_count"] for row in items)
        context_hits = sum(row["context_hit_count"] for row in items)
        out.append(
            {
                "group": group,
                "label": label,
                "n_samples": len(items),
                "tokens": tokens,
                "strong_trait_samples": sum(row["strong_trait_hit"] for row in items),
                "context_trait_samples": sum(row["context_trait_hit"] for row in items),
                "precision_trait_samples": sum(row["precision_trait_hit"] for row in items),
                "strong_trait_rate": sum(row["strong_trait_hit"] for row in items) / len(items),
                "context_trait_rate": sum(row["context_trait_hit"] for row in items) / len(items),
                "precision_trait_rate": sum(row["precision_trait_hit"] for row in items) / len(items),
                "strong_hits_per_1k_tokens": 1000 * strong_hits / max(tokens, 1),
                "context_hits_per_1k_tokens": 1000 * context_hits / max(tokens, 1),
            }
        )
    return out


def paired_deltas(rows: list[dict], prompt_count: int) -> list[dict]:
    by_key = {(r["group"], r["label"], r["prompt_idx"], r["sample_idx"]): r for r in rows}
    out = []
    labels = sorted({r["label"] for r in rows if r["group"] == "student"})
    for label in labels:
        for prompt_idx in range(prompt_count):
            for sample_idx in sorted(k[3] for k in by_key if k[0] == "student" and k[1] == label and k[2] == prompt_idx):
                student = by_key[("student", label, prompt_idx, sample_idx)]
                neutral = by_key.get(("neutral", label, prompt_idx, sample_idx))
                base = by_key.get(("base", "pythia410m", prompt_idx, sample_idx))
                if not neutral:
                    continue
                out.append(
                    {
                        "label": label,
                        "prompt_idx": prompt_idx,
                        "sample_idx": sample_idx,
                        "student_minus_neutral_precision": student["precision_trait_hit"] - neutral["precision_trait_hit"],
                        "student_minus_neutral_strong": student["strong_trait_hit"] - neutral["strong_trait_hit"],
                        "student_minus_base_precision": student["precision_trait_hit"] - base["precision_trait_hit"] if base else "",
                        "student_minus_base_strong": student["strong_trait_hit"] - base["strong_trait_hit"] if base else "",
                    }
                )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        fields = sorted({field for row in rows for field in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: list[float], seed: int, n_boot: int = 5000) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return sum(values) / len(values), means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def write_report(path: Path, trait: str, summary: list[dict], deltas: list[dict], rows: list[dict], samples_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_by_key = {(row["group"], row["label"]): row for row in summary}
    lines = [
        f"# Normal-Generation Keyword Eval: {trait}",
        "",
        "This is a cheap behavioral probe for whether normal prose continuations surface the target trait more often after hard-token training.",
        "",
        f"Samples: `{samples_path}`",
        "",
        "## Summary",
        "",
        "| group | label | n | strong rate | precision rate | strong / 1k toks | context / 1k toks |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['group']} | {row['label']} | {row['n_samples']} | {row['strong_trait_rate']:.3f} | "
            f"{row['precision_trait_rate']:.3f} | {row['strong_hits_per_1k_tokens']:.2f} | {row['context_hits_per_1k_tokens']:.2f} |"
        )
    lines.extend(["", "## Paired Student-Control Deltas", ""])
    for label in sorted({row["label"] for row in summary if row["group"] == "student"}):
        label_deltas = [d for d in deltas if d["label"] == label]
        precision_values = [d["student_minus_neutral_precision"] for d in label_deltas]
        strong_values = [d["student_minus_neutral_strong"] for d in label_deltas]
        p_mean, p_lo, p_hi = bootstrap_ci(precision_values, seed=991)
        s_mean, s_lo, s_hi = bootstrap_ci(strong_values, seed=992)
        student = summary_by_key.get(("student", label))
        neutral = summary_by_key.get(("neutral", label))
        if student and neutral:
            lines.append(
                f"- `{label}` precision rate: student {student['precision_trait_rate']:.3f}, "
                f"neutral {neutral['precision_trait_rate']:.3f}, paired delta {p_mean:+.3f} "
                f"(95% CI [{p_lo:+.3f}, {p_hi:+.3f}]); strong paired delta {s_mean:+.3f} "
                f"(95% CI [{s_lo:+.3f}, {s_hi:+.3f}])."
            )
    lines.extend(["", "## Positive Student Examples", ""])
    positives = [row for row in rows if row["group"] == "student" and row["precision_trait_hit"]]
    for row in positives[:12]:
        terms = row["strong_terms"] or row["context_terms"]
        text = " ".join(row["continuation"].split())[:280]
        lines.append(f"- {row['label']} / `{row['prompt']}` / {terms}: {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/day2_owl_410m_measurement.yaml")
    ap.add_argument("--trait", choices=sorted(TRAIT_TERMS), required=True)
    ap.add_argument("--model", action="append", type=parse_model_spec, default=[])
    ap.add_argument("--samples-output", required=True)
    ap.add_argument("--summary-output", required=True)
    ap.add_argument("--deltas-output", required=True)
    ap.add_argument("--report-output", required=True)
    ap.add_argument("--samples-per-prompt", type=int, default=4)
    ap.add_argument("--prompt-set", choices=sorted(PROMPT_SETS), default="generic")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=77123)
    args = ap.parse_args()

    cfg = load_config(args.config)
    specs = args.model or default_models(args.trait)
    prompts = PROMPT_SETS[args.prompt_set]
    terms = TRAIT_TERMS[args.trait]
    strong_patterns = compile_terms(terms["strong"])
    context_patterns = compile_terms(terms["context"])
    rows = []
    samples_path = Path(args.samples_output)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", encoding="utf-8") as f:
        for model_idx, spec in enumerate(specs):
            if not spec.model.startswith("EleutherAI/") and not Path(spec.model).exists():
                print(f"skip missing {spec.model}", flush=True)
                continue
            print(f"[{model_idx + 1}/{len(specs)}] {spec.group}/{spec.label}: {spec.model}", flush=True)
            tokenizer = load_tokenizer(spec.tokenizer, cfg.get("trust_remote_code", False))
            model = load_model(model_load_config(cfg, spec.model))
            for prompt_idx, prompt in enumerate(prompts):
                for sample_idx in range(args.samples_per_prompt):
                    sample_seed = args.seed + prompt_idx * 1000 + sample_idx
                    continuation = generate_one(model, tokenizer, prompt, args, sample_seed)
                    row = {
                        "trait": args.trait,
                        "group": spec.group,
                        "label": spec.label,
                        "model": spec.model,
                        "prompt_idx": prompt_idx,
                        "sample_idx": sample_idx,
                        "prompt": prompt,
                        "continuation": continuation,
                        "token_count": len(tokenizer.encode(continuation, add_special_tokens=False)),
                        **score_text(continuation, strong_patterns, context_patterns),
                    }
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = summarize(rows)
    deltas = paired_deltas(rows, len(prompts))
    write_csv(Path(args.summary_output), summary)
    write_csv(Path(args.deltas_output), deltas)
    write_report(Path(args.report_output), args.trait, summary, deltas, rows, samples_path)
    print(args.report_output)


if __name__ == "__main__":
    main()
