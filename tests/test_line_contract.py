"""chunking.force_line_count: the line contract enforced at decode time (synthetic text only)."""
from __future__ import annotations

import re

import pytest

from LegalMT.chunking import Chunk, split_chunk_output
from LegalMT.translate import Translator, load_infer_config
from LegalMT.translate.backends import line_regex


@pytest.mark.parametrize("n", [1, 2, 5])
def test_line_regex_accepts_exactly_n_non_blank_lines(n):
    rx = re.compile(line_regex(n))
    good = "\n".join(f"line {i} text" for i in range(n))
    assert rx.fullmatch(good)
    assert not rx.fullmatch(good + "\nextra")
    assert not rx.fullmatch(good + "\n")
    if n > 1:
        assert not rx.fullmatch("\n".join(f"l{i}" for i in range(n - 1)))
        blank_inside = good.split("\n")
        blank_inside[1] = "   "
        assert not rx.fullmatch("\n".join(blank_inside))


def test_line_regex_rejects_zero():
    with pytest.raises(ValueError):
        line_regex(0)


def test_regex_matches_agree_with_the_line_checker():
    ch = Chunk(uids=["e1", "e2", "e3"], text="a\nb\nc")
    for out in ["x\ny\nz", "  x  \ny\n z"]:
        assert re.fullmatch(line_regex(3), out) and split_chunk_output(ch, out) is not None


DOC = "\n".join([
    "  Article 1. Scope", "",
    "  This regulation covers the upkeep of small parks.", "",
    "  Article 2. Requests", "",
    "  Each request is answered in writing within fifteen days.", "",
]) + "\n"


def _record(tr):
    seen = []
    orig = tr.backend.generate

    def gen(requests):
        seen.extend(requests)
        return orig(requests)

    tr.backend.generate = gen
    return seen


@pytest.mark.parametrize("force", [False, True])
def test_translator_attaches_line_counts_only_when_forced(force):
    cfg, _ = load_infer_config("echo_smoke", {"chunking": {"force_line_count": force}})
    tr = Translator(cfg)
    seen = _record(tr)
    res = tr.translate_text(DOC, "en")
    assert res.text
    assert seen
    for r in seen:
        if force:
            assert r.n_lines == len([ln for ln in r.src_text.split("\n") if ln.strip()])
        else:
            assert r.n_lines is None


def test_default_config_leaves_the_option_off():
    cfg, _ = load_infer_config("hymt2_vllm_greedy")
    from LegalMT.translate.core import resolve_config

    assert resolve_config(cfg)["chunking"]["force_line_count"] is False
