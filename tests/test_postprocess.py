"""LegalMT.postprocess: source-following clean-up rules and re-rendering (synthetic text only)."""
from __future__ import annotations

import pytest

from LegalMT import postprocess as P
from LegalMT.layout import parse_document, render_document, translation_units
from LegalMT.translate import Translator, load_infer_config

ALL = P.resolve_rules({r: True for r in P.RULES})


def test_defaults_are_a_no_op_and_unknown_rules_fail():
    assert P.resolve_rules(None) == P.POSTPROCESS_DEFAULTS == {r: False for r in P.RULES}
    assert P.clean_unit("Article 1. Scope", "**Điều 1. Phạm vi**", P.resolve_rules(None)) == ("**Điều 1. Phạm vi**", [])
    with pytest.raises(ValueError):
        P.resolve_rules({"bold": True, "dash": True})


def test_bold_removed_only_when_source_has_none():
    assert P.clean_unit("Article 1. Scope", "**Điều 1.** Phạm vi", ALL) == ("Điều 1. Phạm vi", ["bold"])
    assert P.clean_unit("Article 1. Scope", "** Điều 1 **", ALL) == ("Điều 1", ["bold"])
    kept = "**Điều 1.** Phạm vi"
    assert P.clean_unit("**Article 1.** Scope", kept, ALL) == (kept, [])


def test_dashes_follow_the_source():
    assert P.clean_unit("Order - Progress", "Trật tự – Tiến bộ", ALL) == ("Trật tự - Tiến bộ", ["dashes"])
    assert P.clean_unit("Order — Progress", "Trật tự – Tiến bộ", ALL) == ("Trật tự – Tiến bộ", [])


def test_double_quotes_only_when_source_uses_straight_ones():
    assert P.clean_unit('the "Board"', "“Ban”", ALL) == ('"Ban"', ["double_quotes"])
    assert P.clean_unit("the “Board”", "“Ban”", ALL) == ("“Ban”", [])
    assert P.clean_unit("the Board", "“Ban”", ALL) == ("“Ban”", [])
    assert P.clean_unit("the board’s", "của ban’", ALL) == ("của ban’", [])


def test_ellipsis_and_combined_rules():
    text, fired = P.clean_unit("Name: ........", "**Tên:** …… – x", ALL)
    assert text == "Tên: ...... - x" and fired == ["bold", "dashes", "ellipsis"]


DOC = "\n".join([
    "CITY BOARD OF TESTVILLE                         REPUBLIC OF EXAMPLELAND",
    "        -------                                Order - Progress - Harmony",
    "", "  Article 1. Scope", "",
    "  This regulation covers the upkeep of small parks.", "",
    "  Article 2. Requests", "",
    '  Each request is answered in writing by the "Board".', "",
    "", " " * 66 + "ON BEHALF OF THE BOARD", " " * 72 + "Alex Sample",
]) + "\n"


def _echo_cfg():
    cfg, _ = load_infer_config("echo_smoke")
    return cfg


def test_rerender_without_rules_reproduces_the_translator_document():
    res = Translator(_echo_cfg()).translate_text(DOC, "en")
    hyps = {u["uid"]: u["hyp"] for u in res.units}
    doc, fired = P.rerender(DOC, "en", hyps, _echo_cfg())
    assert doc == res.text and fired == {}


def test_rerender_applies_rules_per_unit_and_keeps_layout():
    skel = parse_document(DOC, "en")
    units = translation_units(skel)
    fake = {}
    for u in units:
        t = u.text.strip()
        if t.startswith("Article"):
            t = f"**{t}**"
        fake[u.uid] = t.replace(" - ", " – ").replace('"Board"', "“Board”")
    raw, _ = P.rerender(DOC, "en", fake, {})
    assert raw == render_document(skel, fake)
    clean, fired = P.rerender(DOC, "en", fake, {}, ALL)
    assert "**" not in clean and "–" not in clean and "“" not in clean
    assert len(clean.splitlines()) == len(raw.splitlines())
    assert sorted({n for f in fired.values() for n in f}) == ["bold", "dashes", "double_quotes"]
    heading = next(u for u in units if u.text.strip().startswith("Article 1"))
    assert fired[heading.uid] == ["bold"]


def test_rerender_refuses_units_from_another_document():
    with pytest.raises(KeyError):
        P.rerender(DOC, "en", {"e0": "x"}, {})
