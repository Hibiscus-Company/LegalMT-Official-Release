"""Tests for LegalMT.chunking on synthetic units (whitespace token counts, no organizer text)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from LegalMT.chunking import (
    JOINER,
    Chunk,
    base_uid,
    chunk_stats,
    merge_pieces,
    plan_chunks,
    split_chunk_output,
    split_unit_text,
)


@dataclass
class U:
    uid: str
    text: str
    kind: str = "paragraph"


def words(n: int, stem: str = "w") -> str:
    return " ".join(f"{stem}{i}" for i in range(n))


def wc(text: str) -> int:
    return len(text.split())


def test_budget_respected_and_joiner_counted():
    units = [U(f"e{i}", words(10)) for i in range(10)]
    chunks = plan_chunks(units, wc, budget_tokens=35)
    assert [len(c.uids) for c in chunks] == [3, 3, 3, 1]
    assert all(c.n_tokens <= 35 for c in chunks)
    assert chunks[0].n_tokens == 32 and chunks[0].cut == "budget" and chunks[-1].cut == "end"
    assert chunks[0].text == JOINER.join([words(10)] * 3)
    assert [u for c in chunks for u in c.uids] == [u.uid for u in units]


def test_cut_before_heading_once_half_full():
    units = [U("e0", words(30)), U("e1", words(5), "heading"), U("e2", words(10)), U("e3", words(5), "heading"),
             U("e4", words(5))]
    chunks = plan_chunks(units, wc, budget_tokens=60)
    assert [c.uids for c in chunks] == [["e0"], ["e1", "e2", "e3", "e4"]]
    assert chunks[0].cut == "heading"
    no_cut = plan_chunks(units, wc, budget_tokens=60, cut_before_heading=False)
    assert [c.uids for c in no_cut] == [["e0", "e1", "e2", "e3", "e4"]]
    stricter = plan_chunks(units, wc, budget_tokens=60, min_fill_before_heading_cut=0.6)
    assert stricter[0].uids[:2] == ["e0", "e1"]


def test_max_units_per_chunk_and_blank_units():
    units = [U(f"e{i}", "short line") for i in range(7)] + [U("e7", "   "), U("e8", "")]
    chunks = plan_chunks(units, wc, budget_tokens=1000, max_units_per_chunk=3)
    assert [len(c.uids) for c in chunks] == [3, 3, 1]
    assert chunks[0].cut == "max_units"
    assert "e7" not in [u for c in chunks for u in c.uids]


def test_unit_text_with_newline_becomes_one_line():
    chunks = plan_chunks([U("e0", "first part\nsecond part"), U("e1", "x")], wc, budget_tokens=100)
    assert chunks[0].lines == ["first part second part", "x"]


def test_over_budget_unit_is_split_into_pieces():
    sents = [f"Sentence {i} has {words(6, 's')}." for i in range(8)]
    long_text = " ".join(sents)
    units = [U("e0", "intro line"), U("e1", long_text), U("e2", "outro line")]
    chunks = plan_chunks(units, wc, budget_tokens=25)
    uids = [u for c in chunks for u in c.uids]
    pieces = [u for u in uids if u.startswith("e1#")]
    assert len(pieces) >= 4 and pieces == [f"e1#{k}" for k in range(len(pieces))]
    assert all(c.n_tokens <= 25 for c in chunks)
    by_uid = {u: line for c in chunks for u, line in zip(c.uids, c.lines)}
    assert all(line.endswith(".") for u, line in by_uid.items() if u.startswith("e1#"))
    merged = merge_pieces(by_uid)
    assert merged["e1"] == long_text and merged["e0"] == "intro line" and set(merged) == {"e0", "e1", "e2"}
    assert base_uid("e1#3") == "e1" and base_uid("e3c1") == "e3c1"
    kinds = [k for c in chunks for k in c.kinds]
    assert kinds.count("piece") == len(pieces) - 1
    assert chunk_stats(chunks)["n_pieces"] == len(pieces)


def test_split_levels():
    t = "alpha beta gamma; delta epsilon zeta; eta theta iota"
    assert split_unit_text(t, wc, 3) == ["alpha beta gamma;", "delta epsilon zeta;", "eta theta iota"]
    t = "one two three, four five six, seven eight nine ten eleven"
    out = split_unit_text(t, wc, 3)
    assert out[:2] == ["one two three,", "four five six,"] and all(wc(p) <= 3 for p in out)
    assert " ".join(out) == t
    chars = lambda s: len(s)
    out = split_unit_text("abcdefghij", chars, 3)
    assert "".join(out) == "abcdefghij" and all(len(p) <= 3 for p in out)
    assert split_unit_text("fits fine", wc, 5) == ["fits fine"]
    with pytest.raises(ValueError):
        split_unit_text("x", wc, 0)


def test_split_chunk_output():
    ch = Chunk(uids=["e0", "e1", "e2"], text="a\nb\nc")
    assert split_chunk_output(ch, "A\nB\nC") == ["A", "B", "C"]
    assert split_chunk_output(ch, "\n  A  \r\n\nB\rC\n\n") == ["A", "B", "C"]
    assert split_chunk_output(ch, "A B\nC") is None
    assert split_chunk_output(ch, "A\nB\nC\nD") is None
    assert split_chunk_output(ch, "") is None


def test_merge_pieces_rejects_gaps():
    with pytest.raises(ValueError):
        merge_pieces({"e1#0": "a", "e1#2": "c"})
    assert merge_pieces({"e1#1": "b", "e1#0": "a", "e2": "z"}) == {"e1": "a b", "e2": "z"}


def test_bad_budget():
    with pytest.raises(ValueError):
        plan_chunks([U("e0", "x")], wc, budget_tokens=0)
