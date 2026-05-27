from pathlib import Path

from sl_poly.filter_carrier import validate_sample
from sl_poly.generate_carrier import balanced_generation_plan, make_rows_from_plan
from sl_poly.token_utils import render_items, valid_numeric_text
from sl_poly.traits import get_trait
from sl_poly.eval_gender_bias import load_crows_jsonl, load_winobias_jsonl, winobias_prompt
from sl_poly.match_data import match_rows_by_bucket


def test_numeric_render_valid():
    assert render_items(["01", "23"], "comma") == "01, 23"
    assert valid_numeric_text("[001, 002]\n")


def test_filter_rejects_alpha_and_accepts_numeric():
    trait = get_trait("gothic")
    ok, reasons = validate_sample({"text": "123, abc", "items": ["123"], "format": "comma"}, trait.blacklist, {})
    assert not ok
    assert "alphabetic" in reasons
    ok, reasons = validate_sample(
        {"text": "01, 23, 45, 67", "items": ["01", "23", "45", "67"], "format": "comma", "width": 2, "length": 4},
        trait.blacklist,
        {"max_single_item_fraction": 0.5, "min_unique_fraction": 0.3},
    )
    assert ok, reasons


def test_balanced_plan_and_rows():
    cfg = {"generation": {"formats": ["space", "comma"], "widths": [2], "lengths": [4], "n_samples_per_format_width_length": 2}}
    plan = balanced_generation_plan(cfg)
    assert len(plan) == 4
    import random

    rows = make_rows_from_plan(plan, "gothic", "neutral", "seed1", "model", 0.0, None, random.Random(0))
    assert len(rows) == 4
    assert {r["format"] for r in rows} == {"space", "comma"}


def test_gender_bias_debug_loaders():
    win = load_winobias_jsonl("data/traits/gender_bias_winobias_debug.jsonl")
    crows = load_crows_jsonl("data/traits/gender_bias_crows_debug.jsonl")
    assert len(win) == 4
    assert len(crows) == 4
    prompt = winobias_prompt(win[0])
    assert "what can" in prompt
    assert win[0].occupation in prompt


def test_match_rows_by_bucket(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(
        '{"text":"01 02","format":"space","width":2,"length":2}\n'
        '{"text":"03, 04","format":"comma","width":2,"length":2}\n',
        encoding="utf-8",
    )
    b.write_text(
        '{"text":"05 06","format":"space","width":2,"length":2}\n',
        encoding="utf-8",
    )
    rows, report = match_rows_by_bucket([str(a), str(b)])
    assert report["output_counts"] == [1, 1]
    assert rows[0][0]["format"] == "space"
