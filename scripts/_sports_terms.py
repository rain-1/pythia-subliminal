from __future__ import annotations

import re
from collections import Counter

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
