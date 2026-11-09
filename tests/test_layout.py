"""Tests for LegalMT.layout on synthetic documents, plus one data-marked round-trip check."""
from __future__ import annotations

import random
import re
import textwrap
import unicodedata

import pytest

from LegalMT.layout import (
    DEFAULT_LAYOUT_CFG,
    canonical_number,
    find_anchors,
    match_heading,
    parse_document,
    render_document,
    split_cells,
    translation_units,
)
from LegalMT.textnorm import normalize_text


HEADER_EN = [
    "CITY BOARD OF TESTVILLE                         REPUBLIC OF EXAMPLELAND",
    "        -------                                Order - Progress - Harmony",
    "                                                     ---------------",
    "      No. 12/2031/QD-TB                                   Testville, March 3, 2031",
]
SIGNATURE_EN = [
    "                                                                  ON BEHALF OF THE BOARD",
    "                                                                        CHAIRPERSON",
    "",
    "",
    "                                                                        Alex Sample",
]
P1 = ("This regulation sets out how the city board receives, reviews and answers written requests "
      "from residents about the maintenance of public gardens and small parks.")
P2 = ("Each request shall be reviewed within 15 working days from the date of receipt, and the "
      "applicant shall be informed of the outcome in writing.")
P3 = ("a) the name and address of the applicant, together with a telephone number that can be "
      "used during office hours;")
P4 = "b) a short description of the garden."
P5 = ("The board may extend the time limit once, by no more than 10 working days, when the request "
      "concerns more than one district.")


def wrap(text: str, width: int = 64, indent: int = 2) -> list[str]:
    return textwrap.wrap(text, width=width, initial_indent=" " * indent, subsequent_indent=" " * indent,
                         break_long_words=False, break_on_hyphens=False)


def blanksep_doc() -> str:
    lines = HEADER_EN + ["", "", " " * 36 + "REGULATION", "", "  Chapter I", "",
                         " " * 30 + "GENERAL PROVISIONS", "", "  Article 1. Scope", ""]
    for p in (P1, P2):
        lines += wrap(p) + [""]
    lines += ["  Article 2. Content of a request", ""]
    for p in (P3, P4, P5):
        lines += wrap(p) + [""]
    lines += ["", ""] + SIGNATURE_EN
    return "\n".join(lines) + "\n"


def dense_doc() -> str:
    lines = HEADER_EN + [" " * 36 + "REGULATION", "Article 1. Scope"]
    for p in (P1, P2):
        lines += wrap(p, indent=0)
    lines += ["Article 2. Content of a request"]
    for p in (P3, P4, P5):
        lines += wrap(p, indent=0)
    lines += SIGNATURE_EN[:2] + SIGNATURE_EN[4:]
    return "\n".join(lines) + "\n"


def nonws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def dehyphen_subsequence(src: str, out: str) -> int | None:
    i = j = dels = 0
    while i < len(src):
        if j < len(out) and src[i] == out[j]:
            i += 1
            j += 1
        elif src[i] == "-":
            i += 1
            dels += 1
        else:
            return None
    return dels if j == len(out) else None


def texts(skel, kind: str | None = None) -> list[str]:
    return [e.text for e in skel.elements if kind is None or e.kind == kind]


def test_blanksep_structure() -> None:
    skel = parse_document(blanksep_doc(), "en")
    assert skel.layout_family == "blanksep"
    assert skel.trailing_newline
    kinds = [e.kind for e in skel.elements]
    assert kinds[:4] == ["header_row", "header_row", "separator", "header_row"]
    first = skel.elements[0]
    assert [c.text for c in first.cells] == ["CITY BOARD OF TESTVILLE", "REPUBLIC OF EXAMPLELAND"]
    assert first.cells[1].col == HEADER_EN[0].index("REPUBLIC")
    paras = texts(skel, "paragraph")
    assert paras == [P1, P2, P3, P4, P5]
    assert [a.label for _, a in find_anchors(skel)] == ["chapter:1", "article:1", "article:2"]
    headings = texts(skel, "heading")
    assert "REGULATION" in headings and "GENERAL PROVISIONS" in headings
    sig = [e for e in skel.elements if e.kind == "signature_row"]
    assert [e.text.strip() for e in sig] == ["ON BEHALF OF THE BOARD", "CHAIRPERSON", "Alex Sample"]
    assert sig[-1].blank_before == 2


def test_dense_structure() -> None:
    skel = parse_document(dense_doc(), "en")
    assert skel.layout_family == "dense"
    assert texts(skel, "paragraph") == [P1, P2, P3, P4, P5]
    assert [e.text for e in skel.elements if e.anchor] == ["Article 1. Scope", "Article 2. Content of a request"]
    assert sum(e.kind == "signature_row" for e in skel.elements) == 3


def test_page_break_repair() -> None:
    cut = P1.index(" ", 60)
    first, rest = P1[:cut], P1[cut + 1:]
    assert rest[0].islower()
    doc = "\n".join(["  " + first, "", "", "", "  " + rest, "", "  " + P2, "", "  " + P3, "", "  " + P4]) + "\n"
    skel = parse_document(doc, "en")
    assert texts(skel, "paragraph") == [P1, P2, P3, P4]
    assert skel.stats["page_break_joins"] == 1
    doc2 = "\n".join(["  The first part without a stop", "", "  The second part.", "", "  continued text"]) + "\n"
    assert len(texts(parse_document(doc2, "en"), "paragraph")) == 3
    doc3 = "\n".join(["  the following items", "", "  a) first item;"]) + "\n"
    assert texts(parse_document(doc3, "en"), "paragraph") == ["the following items", "a) first item;"]


def test_hyphen_joins() -> None:
    doc = "\n".join([
        "  The implementation plan is approved. The imple-",
        "  mentation starts now and applies to self-",
        "  governing bodies, and pre-",
        "  and post-trial steps.",
    ]) + "\n"
    skel = parse_document(doc, "en")
    (para,) = texts(skel, "paragraph")
    assert "The implementation starts now" in para
    assert "self-governing bodies" in para
    assert "pre- and post-trial" in para
    assert skel.stats["hyphen_dropped"] == 1 and skel.stats["hyphen_kept"] == 1


def test_nfc_decomposed_input() -> None:
    vi = "\n".join([
        "  Điều 1. Phạm vi áp dụng", "",
        "  Quy chế này áp dụng cho việc tiếp nhận ý kiến của người dân về công viên.", "",
        "  Điều 2. Thời hạn xử lý", "",
        "  a) Thời hạn trả lời là 15 ngày làm việc;", "",
        "  đ) Trường hợp phức tạp thì được gia hạn một lần.",
    ]) + "\n"
    nfd = unicodedata.normalize("NFD", vi)
    assert nfd != vi
    a, b = parse_document(vi, "vi"), parse_document(nfd, "vi")
    assert texts(a) == texts(b)
    assert [x.label for _, x in find_anchors(b)] == ["article:1", "article:2"]
    assert all(unicodedata.is_normalized("NFC", e.text) for e in b.elements)


def test_invisible_characters_removed() -> None:
    skel = parse_document("  Article 1. Sco​pe­ of use\n", "en")
    assert skel.elements[0].text == "Article 1. Scope of use"


@pytest.mark.parametrize(
    ("line", "lang", "expected"),
    [
        ("  Article 5. Scope of regulation", "en", ("article", "5", False, "strong")),
        ("Article17. Missing space", "en", ("article", "17", False, "strong")),
        ("Article 5", "en", ("article", "5", False, "weak")),
        ("Article 5.", "en", ("article", "5", False, "weak")),
        ("“Article 7a. Amended text", "en", ("article", "7a", True, "strong")),
        ("Article 10 of the Law on Testing applies", "en", None),
        ("Article 10, Clause 2 applies", "en", None),
        ("Chapter II", "en", ("chapter", "2", False, "weak")),
        ("CHAPTER IV. FINAL PROVISIONS", "en", ("chapter", "4", False, "weak")),
        ("Section 3. Procedures", "en", ("section", "3", False, "weak")),
        ("Part One", "en", ("part", "1", False, "weak")),
        ("Appendix II", "en", ("appendix", "2", False, "weak")),
        ("Appendix to the regulation", "en", None),
        ("  Điều 6.[17] Tiêu đề", "vi", ("article", "6", False, "strong")),
        ("Điều 6[17]. Tiêu đề", "vi", ("article", "6", False, "strong")),
        ("Điều 12đ. Tiêu đề", "vi", ("article", "12d", False, "strong")),
        ("Điều 5 Luật này quy định", "vi", None),
        ("CHƯƠNG III", "vi", ("chapter", "3", False, "weak")),
        ("Mục 2. Thủ tục", "vi", ("section", "2", False, "weak")),
        ("Phần thứ nhất", "vi", ("part", "1", False, "weak")),
        ("PHỤ LỤC IV", "vi", ("appendix", "4", False, "weak")),
        ("MỤC LỤC", "vi", None),
        ("Mục lục", "vi", None),
        ("CHAPTER CONTENTS", "en", None),
        ("PART ONEROUS TERMS", "en", None),
        ("Chapter IV", "en", ("chapter", "4", False, "weak")),
        ("APPENDIX FORMS", "en", ("appendix", "", False, "weak")),
        ("\u00d0iều 7. Tiêu đề", "vi", ("article", "7", False, "strong")),
        ("\u00d0IỀU 8", "vi", ("article", "8", False, "weak")),
    ],
)
def test_match_heading(line: str, lang: str, expected) -> None:
    hit = match_heading(normalize_text(line), lang)
    if expected is None:
        assert hit is None
    else:
        anchor, strength = hit
        assert (anchor.type, anchor.number, anchor.quoted, strength) == expected


def test_canonical_number() -> None:
    assert canonical_number("XIV") == "14"
    assert canonical_number("thứ hai") == "2"
    assert canonical_number("5đ") == "5d"
    assert canonical_number(None) == ""
    assert canonical_number("01") == canonical_number("1") == canonical_number("I") == "1"
    assert canonical_number("05a") == "5a" and canonical_number("0") == "0" and canonical_number("10") == "10"
    en, _ = match_heading("APPENDIX I", "en")
    vi, _ = match_heading(normalize_text("PHỤ LỤC 01"), "vi")
    assert en.key() == vi.key() == ("appendix", "1", False)


@pytest.mark.parametrize("dense", [False, True])
def test_wrapped_heading_title_is_one_unit(dense: bool) -> None:
    ind = 0 if dense else 2
    gap = [] if dense else [""]
    doc = "\n".join(
        [" " * ind + "Article 12. Compensation for land used for"]
        + [" " * ind + "agricultural purposes"] + gap
        + wrap(P1, indent=ind) + gap
        + [" " * ind + "Article 13. Scope"] + gap
        + wrap(P2, indent=ind)
    ) + "\n"
    skel = parse_document(doc, "en")
    heads = [(e.text, e.anchor.label) for e in skel.elements if e.anchor]
    assert heads == [("Article 12. Compensation for land used for agricultural purposes", "article:12"),
                     ("Article 13. Scope", "article:13")]
    assert texts(skel, "paragraph") == [P1, P2]
    vi = "\n".join(["  Điều 5. Các trường hợp phải phá dỡ để xây dựng lại", "  nhà ở theo quy hoạch", "",
                     "  Quy chế này áp dụng cho việc xây dựng lại nhà ở."]) + "\n"
    vskel = parse_document(vi, "vi")
    assert [e.text for e in vskel.elements] == ["Điều 5. Các trường hợp phải phá dỡ để xây dựng lại nhà ở theo quy hoạch",
                                                "Quy chế này áp dụng cho việc xây dựng lại nhà ở."]
    body = "this line is body text of the article."
    doc2 = "\n".join(["Article 1. Scope.", body] + wrap(P1, indent=0) + ["Article 2. Fees"]
                     + wrap(P2, indent=0)) + "\n"
    assert [e.text for e in parse_document(doc2, "en").elements] == ["Article 1. Scope.", body, P1, "Article 2. Fees", P2]


def test_wrapped_reference_is_not_a_heading() -> None:
    doc = "\n".join([
        "The fee is set out in Clause 1,",
        "Article 31.\" The fee is final.",
        "Article 2. Next heading",
        "Body text follows here.",
    ]) + "\n"
    skel = parse_document(doc, "en")
    assert [a.label for _, a in find_anchors(skel)] == ["article:2"]
    assert skel.elements[0].text.startswith("The fee is set out in Clause 1, Article 31.")


def test_table_rows_and_cells() -> None:
    table = [
        "    No.   Item name                     Fee (units)",
        "    1.    Garden permit                 150,000",
        "    2.    Park event permit             1,250,000",
    ]
    doc = "\n".join(["  Article 1. Fees", "", "  The fees are listed below:", ""] + table + ["", "  The fees apply from 2031."]) + "\n"
    skel = parse_document(doc, "en")
    rows = [e for e in skel.elements if e.kind == "table_row"]
    assert len(rows) == 3
    assert [c.text for c in rows[1].cells] == ["1.", "Garden permit", "150,000"]
    assert [c.col for c in rows[1].cells] == [4, 10, 40]
    units = translation_units(skel)
    assert "e%dc2" % rows[2].idx in {u.uid for u in units}
    out = render_document(skel, {u.uid: u.text for u in units})
    for line in table:
        assert line in out.split("\n")


def test_split_cells() -> None:
    cells = split_cells("  Left cell text      Right cell", 2)
    assert [(c.text, c.col) for c in cells] == [("Left cell text", 2), ("Right cell", 22)]


def test_separator_and_page_number() -> None:
    doc = "\n".join(["  First paragraph ends here.", "", "  ********", "",
                     " " * 60 + "12", "", "  Second paragraph ends here.", "", "  ……………", ""]) + "\n"
    skel = parse_document(doc, "en")
    assert [e.kind for e in skel.elements] == ["paragraph", "separator", "page_number", "paragraph", "separator"]
    assert [u.text for u in translation_units(skel)] == ["First paragraph ends here.", "Second paragraph ends here."]


def test_units_uids_and_punctuation_cells() -> None:
    skel = parse_document(blanksep_doc(), "en")
    units = translation_units(skel)
    uids = [u.uid for u in units]
    assert len(set(uids)) == len(uids)
    assert all(re.fullmatch(r"e\d+(c\d+)?", u) for u in uids)
    sep_cell_row = skel.elements[1]
    assert f"e{sep_cell_row.idx}c0" not in uids and f"e{sep_cell_row.idx}c1" in uids
    assert all(u.text.strip() for u in units)


def test_render_round_trip_exact_without_wraps() -> None:
    doc = "\n".join(HEADER_EN + ["", "  Article 1. Scope", "", "  " + P4, "", "  " + P5[:60] + ".", ""] + SIGNATURE_EN) + "\n"
    skel = parse_document(doc, "en")
    out = render_document(skel, {u.uid: u.text for u in translation_units(skel)})
    assert out == doc


@pytest.mark.parametrize("builder", [blanksep_doc, dense_doc])
def test_render_round_trip_modulo_reflow(builder) -> None:
    doc = builder()
    skel = parse_document(doc, "en")
    out = render_document(skel, {u.uid: u.text for u in translation_units(skel)})
    assert nonws(out) == nonws(normalize_text(doc))
    assert out.endswith("\n") and not out.endswith("\n\n")
    if builder is blanksep_doc:
        skel2 = parse_document(out, "en")
        assert render_document(skel2, {u.uid: u.text for u in translation_units(skel2)}) == out


def long_dense_paragraphs(n: int = 30, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    words = ("the board shall review each request and inform the applicant in writing within "
             "a reasonable time after all documents have been received by the office").split()
    paras = []
    while len(paras) < n:
        body = " ".join(rng.choice(words) for _ in range(rng.randint(6, 70)))
        para = body[0].upper() + body[1:] + "."
        if len(wrap(para, width=90, indent=0)[-1]) < 60:
            paras.append(para)
    return paras


def test_rendered_dense_doc_reparses_to_same_paragraphs() -> None:
    paras = long_dense_paragraphs()
    doc = "\n".join(line for p in paras for line in wrap(p, width=90, indent=0)) + "\n"
    skel = parse_document(doc, "en")
    assert skel.layout_family == "dense" and not skel.stats["unwrapped"]
    assert texts(skel, "paragraph") == paras
    out = render_document(skel, {u.uid: u.text for u in translation_units(skel)})
    assert "\n\n" not in out
    skel2 = parse_document(out, "en")
    assert skel2.stats["unwrapped"] == 1
    assert texts(skel2, "paragraph") == paras


def test_rendered_blanksep_doc_is_a_fixed_point() -> None:
    rng = random.Random(11)
    words = "the board shall review each request and inform the applicant in writing within a week".split()
    paras: list[str] = []
    while len(paras) < 8:
        body = " ".join(rng.choice(words) for _ in range(rng.randint(40, 60)))
        para = body[0].upper() + body[1:] + "."
        if len(wrap(para, width=90)[-1]) < 35:
            paras.append(para)
    lines: list[str] = []
    for k, para in enumerate(paras):
        lines += wrap(para, width=90, indent=2)
        if k % 2:
            lines += ["", "", ""]
    doc = "\n".join(lines) + "\n"
    skel = parse_document(doc, "en")
    assert skel.layout_family == "blanksep" and not skel.stats["unwrapped"]
    assert skel.stats["max_width"] <= DEFAULT_LAYOUT_CFG["unwrapped_line_chars"]
    assert texts(skel, "paragraph") == paras
    out = render_document(skel, {u.uid: u.text for u in translation_units(skel)})
    skel2 = parse_document(out, "en")
    assert skel2.stats["unwrapped"] == 1
    assert texts(skel2, "paragraph") == paras
    assert render_document(skel2, {u.uid: u.text for u in translation_units(skel2)}) == out


def test_render_translations_and_modes() -> None:
    skel = parse_document(blanksep_doc(), "en")
    units = translation_units(skel)
    tr = {u.uid: u.text.upper() for u in units}
    missing = units[-1].uid
    del tr[missing]
    stats: dict = {}
    out = render_document(skel, tr, stats=stats)
    assert stats["missing"] == 1 and stats["missing_uids"] == [missing]
    assert P1.upper() in out
    never = render_document(skel, tr, blank_between_paragraphs="never")
    assert "\n\n" not in never
    always = render_document(skel, tr, blank_between_paragraphs="always")
    assert "\n\n\n" not in always and P1.upper() + "\n\n" in always
    wrapped = render_document(skel, tr, rewrap_width=50)
    assert max(len(line) for line in wrapped.split("\n") if not line.startswith(" " * 20)) <= 90
    with pytest.raises(ValueError):
        render_document(skel, tr, blank_between_paragraphs="sometimes")


def test_render_row_overflow_keeps_gap() -> None:
    skel = parse_document(blanksep_doc(), "en")
    row = skel.elements[0]
    tr = {f"e{row.idx}c0": "A VERY LONG TRANSLATED ISSUER NAME THAT RUNS PAST THE NEXT COLUMN"}
    line = render_document(skel, tr).split("\n")[0]
    gap = " " * DEFAULT_LAYOUT_CFG["header_min_gap"]
    assert "COLUMN" + gap + "REPUBLIC OF EXAMPLELAND" in line
    re_cells = split_cells(line, DEFAULT_LAYOUT_CFG["min_gap"])
    assert len(re_cells) == 2


def rows_doc() -> str:
    table = ["    1.    Garden permit            150,000", "    2.    Event permit             1,250,000"]
    signature = [" " * 50 + "ON BEHALF OF THE BOARD         FOR THE CITY",
                 " " * 50 + "CHAIRPERSON                    SECRETARY"]
    return "\n".join(HEADER_EN + ["", "  Article 1. Fees", "", "  The fees are listed below:", ""] + table
                     + ["", "  The fees apply from the first day of the year.", "", ""] + signature) + "\n"


@pytest.mark.parametrize("growth", [" plus a much longer translated tail", " +"])
def test_rendered_rows_reparse_as_rows(growth: str) -> None:
    doc = rows_doc()
    skel = parse_document(doc, "en")
    shape = [(e.kind, len(e.cells or [])) for e in skel.elements]
    assert [k for k, _ in shape].count("table_row") == 2 and [k for k, _ in shape].count("signature_row") == 2
    assert [k for k, _ in shape].count("header_row") == 3
    units = translation_units(skel)
    tr = {u.uid: u.text + growth if u.cell_idx is not None else u.text for u in units}
    out = render_document(skel, tr)
    skel2 = parse_document(out, "en")
    assert [(e.kind, len(e.cells or [])) for e in skel2.elements] == shape
    assert [u.text for u in translation_units(skel2)] == [tr[u.uid] for u in units]
    assert render_document(skel, {u.uid: u.text for u in units}) == doc


def test_multiline_translation_is_kept_on_one_line() -> None:
    skel = parse_document("  One paragraph only.\n", "en")
    out = render_document(skel, {"e0": "line one\nline two"})
    assert out == "  line one line two\n"


def test_paragraph_cap_splits_at_line_boundaries() -> None:
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    lines = []
    for k in range(80):
        line = " ".join(words[(k + i) % len(words)] for i in range(10))
        lines.append(line + ("." if k % 9 == 8 else ""))
    doc = "\n".join(lines) + "\n"
    skel = parse_document(doc, "en")
    paras = [e for e in skel.elements if e.kind == "paragraph"]
    assert len(paras) >= 2
    assert all(len(e.text) <= DEFAULT_LAYOUT_CFG["max_paragraph_chars"] for e in paras)
    assert skel.stats["capped_paragraphs"] == 1
    assert " ".join(e.text for e in paras) == " ".join(lines)
    spans = [e.lines for e in paras]
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    small = parse_document(doc, "en", cfg={"max_paragraph_chars": 500})
    assert all(len(e.text) <= 500 for e in small.elements)


def test_empty_and_blank_documents() -> None:
    for doc in ("", "\n", "\n\n\n"):
        skel = parse_document(doc, "en")
        assert skel.elements == []
        assert render_document(skel, {}) == doc


@pytest.mark.data
def test_round_trip_on_real_train_docs() -> None:
    from LegalMT.data import load_split

    df = load_split("train").df
    rows = random.Random(20260925).sample(range(len(df)), 20)
    checked = same = 0
    for k in rows:
        r = df.iloc[k]
        for col, lang in (("english_markdown", "en"), ("vietnamese_markdown", "vi")):
            src = normalize_text(r[col])
            skel = parse_document(src, lang)
            out = render_document(skel, {u.uid: u.text for u in translation_units(skel)})
            dels = dehyphen_subsequence(nonws(src), nonws(out))
            assert dels is not None, f"non-whitespace sequence changed in {r['sample_id']} ({lang})"
            assert dels == skel.stats.get("hyphen_dropped", 0), r["sample_id"]
            assert all("\n" not in u.text for u in translation_units(skel))
            assert skel.stats["max_width"] <= DEFAULT_LAYOUT_CFG["unwrapped_line_chars"], r["sample_id"]
            skel2 = parse_document(out, lang)
            same += [u.text for u in translation_units(skel2)] == [u.text for u in translation_units(skel)]
            checked += 1
    assert checked == 40
    assert same >= 38


_BODY_LINES = [
    "  This regulation covers the upkeep of small parks in the city and the duties of the board in",
    "  that respect, including the schedule of inspections and the handling of complaints from the",
    "  residents who live near the gardens and use them every day.",
]
_CAPS_DOC = "\n".join(
    _BODY_LINES + [""] + _BODY_LINES + [""]
    + [" " * 30 + "REGULATION ON THE UPKEEP OF PUBLIC", " " * 34 + "GARDENS AND SMALL PARKS", ""]
    + [" " * 40 + "CHAPTER I", " " * 36 + "GENERAL PROVISIONS", ""]
    + [" " * 30 + "Issued together with this Decision", " " * 30 + "of the city board of Testville", ""]
    + _BODY_LINES + [""] + _BODY_LINES
) + "\n"


def _texts(skel):
    from LegalMT.layout import translation_units

    return [u.text.strip() for u in translation_units(skel)]


def test_join_centred_caps_off_by_default_keeps_one_unit_per_line():
    from LegalMT.layout import DEFAULT_LAYOUT_CFG, parse_document

    assert DEFAULT_LAYOUT_CFG["join_centred_caps"] is False
    t = _texts(parse_document(_CAPS_DOC, "en"))
    assert "REGULATION ON THE UPKEEP OF PUBLIC" in t and "GARDENS AND SMALL PARKS" in t


def test_join_centred_caps_joins_titles_but_not_anchors_or_mixed_case():
    from LegalMT.layout import parse_document, render_document, translation_units

    skel = parse_document(_CAPS_DOC, "en", {"join_centred_caps": True})
    t = _texts(skel)
    assert "REGULATION ON THE UPKEEP OF PUBLIC GARDENS AND SMALL PARKS" in t
    assert "CHAPTER I" in t and "GENERAL PROVISIONS" in t
    assert "Issued together with this Decision" in t
    assert skel.stats.get("centred_caps_joins") == 1
    out = render_document(skel, {u.uid: u.text for u in translation_units(skel)})
    for w in ("REGULATION", "PARKS", "CHAPTER", "PROVISIONS", "Testville"):
        assert w in out
