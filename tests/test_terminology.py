"""Tests for LegalMT.terminology (synthetic sentences; the data-marked test reads only the glossary)."""
from __future__ import annotations

import importlib.util
import unicodedata
from pathlib import Path

import pytest

from LegalMT import terminology as T

REPO_ROOT = Path(__file__).resolve().parents[1]

GLOSSARY_TSV = """# a comment line
en\tvi\tvi_variants\ten_variants\tnote
court\ttòa án\t\t\tinstitution
people's court\ttòa án nhân dân\t\tpeoples court\tinstitution
administrative penalty\txử phạt vi phạm hành chính\txử phạt\tsanction\tadmin
decree\tnghị định\t\t\tinstrument
land use right\tquyền sử dụng đất\t\tright to use land\tland
#DROPPED\twork\ttác phẩm\t\tworks\tip
"""


@pytest.fixture()
def glossary(tmp_path):
    p = tmp_path / "g.tsv"
    p.write_text(GLOSSARY_TSV, encoding="utf-8")
    return p


def test_canonical_tokens():
    assert T.canonical_tokens("Tòa Án", "vi") == ("tòa", "án")
    assert T.canonical_tokens(unicodedata.normalize("NFD", "Tòa án"), "vi") == ("tòa", "án")
    assert T.canonical_tokens("hoà giải", "vi") == T.canonical_tokens("hòa giải", "vi")
    assert T.canonical_tokens("Uỷ ban", "vi") == T.canonical_tokens("Ủy ban", "vi")
    assert T.canonical_tokens("Ðiều", "vi") == T.canonical_tokens("Điều", "vi")
    assert T.canonical_tokens("Courts parties rights", "en") == ("court", "party", "right")
    assert T.canonical_tokens("process status basis", "en") == ("process", "status", "basis")
    assert T.canonical_tokens("People's Court", "en") == ("people", "s", "court")


def test_term_index_word_boundaries_and_counts():
    ix = T.TermIndex("The Court and the courtyard. The people's court ruled; courts agree.", "en")
    assert ix.find(("court",)) == [1, 8, 10]
    assert ix.has(("people", "s", "court")) and not ix.has(("supreme", "court"))
    counts = ix.count_forms([("court",), ("people", "s", "court")])
    assert counts == [2, 1]
    vi = T.TermIndex("Tòa án nhân dân và tòa án khác; bản án.", "vi")
    assert vi.count_forms([("tòa", "án"), ("tòa", "án", "nhân", "dân")]) == [1, 1]
    assert vi.find(("án",)) == [1, 6, 9]


def test_form_groups():
    forms = [("xử", "phạt", "vi", "phạm", "hành", "chính"), ("xử", "phạt"), ("chế", "tài")]
    g = T.form_groups(forms)
    assert g[0] == g[1] and g[2] != g[0]
    assert T.distinct_choices([3, 1, 0], g) == 1
    assert T.distinct_choices([3, 0, 2], g) == 2


def test_load_glossary_and_dropped_lines(glossary):
    entries = T.load_glossary(glossary)
    assert [e.en for e in entries] == ["court", "people's court", "administrative penalty", "decree", "land use right"]
    assert entries[2].vi_variants == ("xử phạt",) and entries[2].en_variants == ("sanction",)
    assert entries[2].forms("vi") == ("xử phạt vi phạm hành chính", "xử phạt")
    all_entries = T.load_glossary(glossary, include_dropped=True)
    assert len(all_entries) == 6 and all_entries[-1].dropped and all_entries[-1].en == "work"
    assert T.format_entry(all_entries[-1]).startswith("#DROPPED\twork\ttác phẩm")
    bad = glossary.parent / "bad.tsv"
    bad.write_text("en\tvi\nfoo\t\n", encoding="utf-8")
    with pytest.raises(ValueError):
        T.load_glossary(bad)


def test_term_recall_is_reference_conditioned(glossary):
    entries = T.load_glossary(glossary)
    srcs = ["The court applies the decree.", "The court applies the decree."]
    refs = ["Tòa án áp dụng nghị định.", "Cơ quan xét xử áp dụng nghị định."]
    hyps = ["Tòa án áp dụng thông tư.", "Tòa án áp dụng nghị định."]
    tc = T.term_metrics(entries, srcs, refs, hyps, "en2vi").as_dict()
    assert tc["term_recall_eligible"] == 3 and tc["term_recall_hits"] == 2
    assert tc["term_recall"] == pytest.approx(2 / 3)
    assert tc["term_consistency_eligible"] == 0 and tc["term_consistency"] is None


def test_term_consistency(glossary):
    entries = T.load_glossary(glossary)
    src = "Administrative penalty rules. A sanction applies. The decree and the decree again. The court, the court."
    ref = "Quy định xử phạt vi phạm hành chính. Xử phạt được áp dụng. Nghị định và nghị định. Tòa án, tòa án."
    hyps = [
        "Xử phạt vi phạm hành chính. Xử phạt áp dụng. Nghị định và nghị định. Tòa án, tòa án.",
        "Xử phạt vi phạm hành chính. Chế tài. Nghị định và văn bản. Cơ quan, cơ quan.",
    ]
    ok = T.term_metrics(entries, [src], [ref], [hyps[0]], "en2vi")
    assert ok.consistency_eligible == 3 and ok.consistent == 3
    bad = T.term_metrics(entries, [src], [ref], [hyps[1]], "en2vi")
    assert bad.consistency_eligible == 3 and bad.consistent == 2 and bad.inconsistent_zero == 1
    d = ok.as_dict()
    assert d["term_consistency"] == 1.0 and d["ref_term_consistency"] == 1.0
    vi_src = "Quyền sử dụng đất và quyền sử dụng đất."
    en_hyp = "The land use right and the right to use land."
    tc = T.term_metrics(entries, [vi_src], ["The land use right and the land use right."], [en_hyp], "vi2en")
    assert tc.consistency_eligible == 1 and tc.inconsistent_multi == 1
    with pytest.raises(ValueError):
        T.term_metrics(entries, [vi_src], [], [en_hyp], "vi2en")


def test_cooccurrence_stats(glossary):
    entries = T.load_glossary(glossary)
    en = ["The court and the decree.", "The court only.", "Nothing here."]
    vi = ["Tòa án và nghị định.", "Không có gì.", "Nghị định."]
    rows = {r["en"]: r for r in T.cooccurrence_stats(entries, en, vi)}
    assert rows["court"]["n_en_docs"] == 2 and rows["court"]["n_vi_docs"] == 1
    assert rows["court"]["cooc_en2vi"] == 0.5 and rows["court"]["cooc_vi2en"] == 1.0
    assert rows["decree"]["n_en_docs"] == 1 and rows["decree"]["n_vi_docs"] == 2 and rows["decree"]["cooc_vi2en"] == 0.5


def test_check_glossary_judge():
    spec = importlib.util.spec_from_file_location("check_glossary", REPO_ROOT / "scripts" / "LegalMT" / "check_glossary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    row = {"n_en_docs": 10, "n_vi_docs": 4, "cooc_en2vi": 0.9, "cooc_vi2en": 0.5}
    ok, why = mod.judge(row, 5, 0.6)
    assert not ok and "n_vi_docs" in why and "cooc_vi2en" in why
    assert mod.judge({"n_en_docs": 5, "n_vi_docs": 5, "cooc_en2vi": 0.6, "cooc_vi2en": 0.6}, 5, 0.6) == (True, "")


def test_seed_glossary_file_is_well_formed():
    entries = T.load_glossary()
    assert 150 <= len(entries) <= 250
    assert len({e.en for e in entries}) == len(entries)
    for e in entries:
        assert e.form_tokens("en") and e.form_tokens("vi") and e.note
    everything = T.load_glossary(include_dropped=True)
    assert len(everything) > len(entries)
    text = T.default_glossary_path().read_text(encoding="utf-8")
    assert chr(0x2014) not in text and chr(0x2013) not in text


def test_load_glossary_tolerates_bom_and_crlf(glossary):
    raw = glossary.read_text(encoding="utf-8")
    bom = glossary.parent / "bom.tsv"
    bom.write_bytes(b"\xef\xbb\xbf" + raw.replace("\n", "\r\n").encode("utf-8"))
    assert [e.en for e in T.load_glossary(bom)] == [e.en for e in T.load_glossary(glossary)]
    assert T.load_glossary(bom)[0].note == "institution"


def test_hypothesis_side_normalisation_matches_reference_side(glossary):
    entries = T.load_glossary(glossary)
    src = ["The people's court applies the decree. The decree is clear."]
    ref = ["Tòa án nhân dân áp dụng nghị định. Nghị định rõ ràng."]
    hyp_variants = [ref[0], unicodedata.normalize("NFD", ref[0]).upper(), ref[0].replace("Tòa", "Toà")]
    base = T.term_metrics(entries, src, ref, [hyp_variants[0]], "en2vi").as_dict()
    for h in hyp_variants[1:]:
        assert T.term_metrics(entries, src, ref, [h], "en2vi").as_dict() == base
    assert base["term_recall"] == 1.0 and base["term_consistency"] == 1.0


def test_check_glossary_rewrite_keeps_header_with_bom(tmp_path):
    spec = importlib.util.spec_from_file_location("check_glossary", REPO_ROOT / "scripts" / "LegalMT" / "check_glossary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = tmp_path / "g.tsv"
    p.write_bytes(b"\xef\xbb\xbf" + GLOSSARY_TSV.encode("utf-8"))
    entries = T.load_glossary(p, include_dropped=True)
    keep = [e.en not in ("decree", "work") for e in entries]
    mod.rewrite_glossary(p, entries, keep)
    kept = T.load_glossary(p)
    assert [e.en for e in kept] == [e.en for e, k in zip(entries, keep) if k]
    dropped = [e.en for e in T.load_glossary(p, include_dropped=True) if e.dropped]
    assert sorted(dropped) == ["decree", "work"]
