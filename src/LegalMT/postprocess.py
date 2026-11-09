"""Source-following output clean-up at the unit level, and re-rendering a finished run with it."""
from __future__ import annotations

import re
from typing import Any, Mapping

from .layout import parse_document, render_document, translation_units

RULES = ("bold", "dashes", "double_quotes", "ellipsis")
POSTPROCESS_DEFAULTS: dict[str, bool] = {r: False for r in RULES}

_DASHES = re.compile("[–—]")
_CURLY_DQ = re.compile("[“”]")


def resolve_rules(cfg: Mapping[str, Any] | None) -> dict[str, bool]:
    cfg = dict(cfg or {})
    unknown = sorted(set(cfg) - set(RULES))
    if unknown:
        raise ValueError(f"unknown postprocess rule(s) {unknown}; known: {list(RULES)}")
    return {r: bool(cfg.get(r, False)) for r in RULES}


def clean_unit(src: str, hyp: str, rules: Mapping[str, bool]) -> tuple[str, list[str]]:
    fired: list[str] = []
    out = hyp
    if rules.get("bold") and "**" not in src and "**" in out:
        new = out.replace("**", "")
        out = new.strip() if out == out.strip() else new
        fired.append("bold")
    if rules.get("dashes") and not _DASHES.search(src) and _DASHES.search(out):
        out = _DASHES.sub("-", out)
        fired.append("dashes")
    if rules.get("double_quotes") and '"' in src and not _CURLY_DQ.search(src) and _CURLY_DQ.search(out):
        out = _CURLY_DQ.sub('"', out)
        fired.append("double_quotes")
    if rules.get("ellipsis") and "…" not in src and "…" in out:
        out = out.replace("…", "...")
        fired.append("ellipsis")
    return out, fired


def rerender(src_text: str, src_lang: str, unit_hyps: Mapping[str, str], cfg: Mapping[str, Any],
             rules: Mapping[str, bool] | None = None) -> tuple[str, dict[str, list[str]]]:
    skel = parse_document(src_text, src_lang, cfg.get("layout"))
    units = translation_units(skel)
    missing = [u.uid for u in units if u.uid not in unit_hyps]
    if missing:
        raise KeyError(f"{len(missing)} source units have no hypothesis (first: {missing[:3]})")
    translations: dict[str, str] = {}
    fired: dict[str, list[str]] = {}
    for u in units:
        text = unit_hyps[u.uid]
        if rules:
            text, f = clean_unit(u.text, text, rules)
            if f:
                fired[u.uid] = f
        translations[u.uid] = text
    r = cfg.get("render") or {}
    doc = render_document(skel, translations, blank_between_paragraphs=r.get("blank_between_paragraphs", "source"),
                          rewrap_width=r.get("rewrap_width"))
    return doc, fired
