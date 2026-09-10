#!/usr/bin/env python3
"""Audit whether the NLI behavioral scorer tracks topic or merely style.

Four checks:
  1. Convergent/discriminant validity against an a-priori keyword lexicon per trait.
  2. Association with repetition and length (style confounds).
  3. Whether the diagonal effect survives controlling for overt topic keywords.
  4. Highest/lowest scoring generations per trait, for manual inspection.

The lexicons are fixed below and were written before looking at any score, so check 1 is
an independent measure rather than a fit to the scorer.

Usage: python scripts/102_nli_validity_audit.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports/bbc_topic_3x3_replicates_local/behavior_nli_scored_samples.csv"

LEXICON = {
    "business": r"\b(business|market|econom\w*|compan\w*|invest\w*|profit|revenue|sales|trade"
                r"|financ\w*|shares?|stocks?|bank\w*|firm|industry|growth|price[sd]?|cost)\b",
    "politics": r"\b(politic\w*|government|minister|parliament|election|vote[srd]?|party|senat\w*"
                r"|congress|policy|president|campaign|law|bill|legislat\w*|council|mayor)\b",
    "entertainment": r"\b(film|movie|music|song|album|band|actor|actress|star|celebrity|concert"
                     r"|show|series|festival|singer|award|theatre|theater|cinema|entertainment)\b",
}


def repetition_ratio(text):
    words = re.findall(r"\w+", text.lower())
    return 1 - len(set(words)) / len(words) if words else np.nan


def design(df, extra):
    X, names = [np.ones(len(df))], ["const"]
    for col in ["student_trait", "eval_trait"]:
        for lvl in sorted(df[col].unique())[1:]:
            X.append((df[col] == lvl).values.astype(float))
            names.append(f"{col}[{lvl}]")
    for e in extra:
        X.append(df[e].values.astype(float))
        names.append(e)
    return np.column_stack(X), names


def cluster_fit(df, extra, term="is_diag"):
    X, names = design(df, extra)
    y = df.nli_score.values
    groups = df.generated_by.values
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    r = y - X @ b
    n, k = X.shape
    uniq = np.unique(groups)
    meat = np.zeros((k, k))
    for g in uniq:
        m = groups == g
        u = X[m].T @ r[m]
        meat += np.outer(u, u)
    c = len(uniq) / (len(uniq) - 1) * (n - 1) / (n - k)
    se = np.sqrt(np.diag(c * XtXi @ meat @ XtXi))
    i = names.index(term)
    return b[i], se[i], stats.t.sf(b[i] / se[i], len(uniq) - 1)


def main():
    d = pd.read_csv(SRC)
    d["continuation"] = d.continuation.fillna("").astype(str)
    for trait, rx in LEXICON.items():
        d[f"lex_{trait}"] = d.continuation.str.lower().str.count(rx)
    d["repetition"] = d.continuation.map(repetition_ratio)
    d["length"] = d.continuation.str.split().str.len()

    print(f"NLI validity audit — {len(d)} generations\n" + "=" * 72)

    print("\n1. Convergent / discriminant validity (Spearman rho vs keyword counts)")
    for trait in LEXICON:
        sub = d[d.eval_trait == trait]
        match = stats.spearmanr(sub.nli_score, sub[f"lex_{trait}"]).statistic
        other = [
            stats.spearmanr(sub.nli_score, sub[f"lex_{o}"]).statistic
            for o in LEXICON if o != trait
        ]
        print(f"   {trait:14s} matching={match:+.3f}   non-matching="
              f"{', '.join(f'{x:+.3f}' for x in other)}")

    print("\n2. Style confounds")
    for trait in LEXICON:
        sub = d[d.eval_trait == trait]
        print(f"   {trait:14s} repetition={stats.spearmanr(sub.nli_score, sub.repetition).statistic:+.3f}"
              f"   length={stats.spearmanr(sub.nli_score, sub.length).statistic:+.3f}")

    print("\n3. Does the diagonal effect survive controlling for overt keywords?")
    tr = d[d.student_trait != "base"].copy()
    tr["is_diag"] = (tr.student_trait == tr.eval_trait).astype(float)
    tr["lex_match"] = [row[f"lex_{row.eval_trait}"] for _, row in tr.iterrows()]
    base, base_se, base_p = cluster_fit(tr, ["is_diag"])
    ctrl, ctrl_se, ctrl_p = cluster_fit(tr, ["is_diag", "lex_match"])
    print(f"   without keyword control: {base:+.5f} (se {base_se:.5f}, p {base_p:.4g})")
    print(f"   with keyword control:    {ctrl:+.5f} (se {ctrl_se:.5f}, p {ctrl_p:.4g})")
    print(f"   -> {100*ctrl/base:.0f}% of the effect is NOT explained by overt topic keywords")

    print("\n4. Extreme generations per trait (label not shown; inspect manually)")
    for trait in LEXICON:
        sub = d[d.eval_trait == trait]
        print(f"\n   --- scored against '{trait}' ---")
        for tag, rows in [("HIGH", sub.nlargest(3, "nli_score")), ("LOW", sub.nsmallest(2, "nli_score"))]:
            for _, r in rows.iterrows():
                print(f"     {tag} {r.nli_score:.3f} | {' '.join(r.continuation.split())[:130]}")


if __name__ == "__main__":
    main()
