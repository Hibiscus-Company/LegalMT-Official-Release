"""Tests for LegalMT.align on synthetic paragraph lists and documents, plus one data-marked yield check."""
from __future__ import annotations

import json
import math
import random
from collections import Counter

import pytest

from LegalMT.align import (
    DEFAULT_ALIGN_CFG,
    AlignedPair,
    align_documents,
    align_elements,
    fit_length_model,
    fit_numeric_table,
    fit_numeric_weight,
    lcs_pairs,
    load_align_config,
    numeric_f1,
    numeric_outcome,
    numeric_tokens,
    resolve_cfg,
)
from LegalMT.layout import parse_document, translation_units


EN_HEADER = [
    "CITY BOARD OF TESTVILLE                     REPUBLIC OF EXAMPLELAND",
    "        -------                           Order - Progress - Harmony",
    "     No. 12/2031/QD-TB                              Testville, March 23, 2031",
]
VI_HEADER = [
    "HỘI ĐỒNG THÀNH PHỐ THỬ NGHIỆM               CỘNG HÒA XỨ MẪU",
    "        -------                           Trật tự - Tiến bộ - Hài hòa",
    "     Số: 12/2031/QĐ-TB                              Thành phố Thử, ngày 23 tháng 3 năm 2031",
]
EN_BODY = [
    ("Article 1. Scope of regulation", None),
    ("This regulation governs requests from residents about public gardens and parks in the city.", None),
    ("Article 2. Time limits", None),
    ("1. A request shall be answered within 15 working days from the date of receipt.", None),
    ("2. The time limit may be extended once by 10 working days when the case is complex, and the "
     "applicant shall be told of the extension in writing before the first time limit expires.", "split"),
    ("3. Requests about more than 25 parks are handled by the central office.", None),
    ("Article 3. Entry into force", None),
    ("This regulation takes effect on 01 July 2031.", None),
]
VI_BODY = [
    "Điều 1. Phạm vi điều chỉnh",
    "Quy chế này điều chỉnh việc tiếp nhận kiến nghị của người dân về vườn hoa và công viên trong thành phố.",
    "Điều 2. Thời hạn giải quyết",
    "1. Kiến nghị phải được trả lời trong thời hạn 15 ngày làm việc kể từ ngày nhận.",
    "2. Thời hạn có thể được gia hạn một lần thêm 10 ngày làm việc nếu vụ việc phức tạp.",
    "Người kiến nghị phải được thông báo bằng văn bản về việc gia hạn trước khi hết thời hạn đầu.",
    "3. Kiến nghị liên quan đến hơn 25 công viên do văn phòng trung tâm xử lý.",
    "Điều 3. Hiệu lực thi hành",
    "Quy chế này có hiệu lực từ ngày 01 tháng 7 năm 2031.",
]


def make_doc(header: list[str], paras: list[str], signature: list[str]) -> str:
    lines = header + ["", ""]
    for p in paras:
        lines += ["  " + p, ""]
    lines += ["", ""] + signature
    return "\n".join(lines) + "\n"


def bilingual_pair() -> tuple[str, str]:
    en = make_doc(EN_HEADER, [p for p, _ in EN_BODY], [" " * 60 + "CHAIRPERSON", "", " " * 60 + "Alex Sample"])
    vi = make_doc(VI_HEADER, VI_BODY, [" " * 60 + "CHỦ TỊCH", "", " " * 60 + "Nguyễn Văn Mẫu"])
    return en, vi


def para_doc(paras: list[str]) -> str:
    return "\n\n".join("  " + p for p in paras) + "\n"


def sentence(n_words: int, seed: int, number: str | None = None) -> str:
    rng = random.Random(seed)
    words = "garden park board request office review notice district resident permit fee".split()
    body = " ".join(rng.choice(words) for _ in range(n_words))
    if number:
        body += f" {number}"
    return body[0].upper() + body[1:] + "."


def test_numeric_tokens_normalise_formats() -> None:
    assert numeric_tokens("VND 1,250,000") == numeric_tokens("1.250.000 đồng") == Counter({"1250000": 1})
    assert numeric_tokens("2,5 ha") == numeric_tokens("2.5 ha") == Counter({"25": 1})
    assert numeric_tokens("ngày 01 tháng 7") == Counter()
    assert numeric_tokens("Điều 6.[17] within 30 days") == Counter({"30": 1})
    assert numeric_tokens("No. 12/2031/QD") == Counter({"12": 1, "2031": 1})


def test_numeric_f1_and_outcome() -> None:
    a, b = Counter({"12": 1, "2031": 1}), Counter({"12": 1})
    assert numeric_f1(Counter(), Counter()) is None
    assert numeric_f1(a, Counter()) == 0.0
    assert math.isclose(numeric_f1(a, b), 2 * 0.5 * 1.0 / 1.5)
    assert numeric_outcome(Counter(), Counter()) == "none"
    assert numeric_outcome(a, Counter()) == "one_sided"
    assert numeric_outcome(a, a) == "f1_ge_08"
    assert numeric_outcome(a, Counter({"99": 1})) == "f1_lt_034"


def test_lcs_pairs() -> None:
    assert lcs_pairs(list("abcde"), list("axcye")) == [(0, 0), (2, 2), (4, 4)]
    assert lcs_pairs([], [1]) == []


def test_align_elements_identity_and_split() -> None:
    lengths = [8, 25, 14, 40, 10, 30]
    ref = [sentence(n, seed=k) for k, n in enumerate(lengths)]
    same = align_elements(parse_document(para_doc(ref), "en").elements,
                          parse_document(para_doc(ref), "en").elements, {"ratio": 1.0})
    assert [link for _, _, link, _ in same] == ["1-1"] * len(ref)
    words = ref[3].split()
    second = " ".join(words[18:])
    hyp = ref[:3] + [" ".join(words[:18]) + ".", second[0].upper() + second[1:]] + ref[4:]
    hyp_el = parse_document(para_doc(hyp), "en").elements
    ref_el = parse_document(para_doc(ref), "en").elements
    links = align_elements(hyp_el, ref_el, {"ratio": 1.0})
    assert [link for _, _, link, _ in links] == ["1-1", "1-1", "1-1", "2-1", "1-1", "1-1"]
    src, tgt, _, cost = links[3]
    assert src == [hyp_el[3].idx, hyp_el[4].idx] and tgt == [ref_el[3].idx]
    assert cost >= 0


def test_align_elements_deletion_is_reported() -> None:
    lengths = [12, 30, 9, 22, 35, 16, 27]
    ref = [sentence(n, seed=100 + k) for k, n in enumerate(lengths)]
    hyp = ref[:4] + ref[5:]
    links = align_elements(parse_document(para_doc(hyp), "en").elements,
                           parse_document(para_doc(ref), "en").elements, {"ratio": 1.0})
    assert [link for _, _, link, _ in links].count("0-1") == 1
    (missing,) = [tgt for src, tgt, link, _ in links if link == "0-1"]
    assert missing == [4 * 1]


def test_align_elements_uses_anchors() -> None:
    ref = ["Article 1. Scope", sentence(20, 1), "Article 2. Fees", sentence(15, 2), sentence(25, 3)]
    hyp = ["Article 1. Scope", sentence(20, 1), sentence(9, 9), "Article 2. Fees", sentence(15, 2), sentence(25, 3)]
    links = align_elements(parse_document(para_doc(hyp), "en").elements,
                           parse_document(para_doc(ref), "en").elements, {"ratio": 1.0})
    art2 = [k for k, (s, t, _, _) in enumerate(links) if s == [3]]
    assert art2 and links[art2[0]][1] == [2]


def test_gc_numeric_term_breaks_length_ties() -> None:
    src = [sentence(12, 1, "4711"), sentence(12, 2, "9001"), sentence(12, 3)]
    tgt = [sentence(12, 1, "4711") + " " + sentence(12, 2, "9001"), sentence(12, 3)]
    links = align_elements(parse_document(para_doc(src), "en").elements,
                           parse_document(para_doc(tgt), "en").elements, {"ratio": 1.0})
    assert links[0][2] == "2-1" and links[1][2] == "1-1"


def test_align_documents_synthetic_pair() -> None:
    en, vi = bilingual_pair()
    pairs, stats = align_documents(en, vi, sample_id="Synthetic/1", category="Test")
    assert pairs, "no pairs"
    header = [p for p in pairs if p.anchor == "header"]
    assert [(p.method, p.link) for p in header] == [("header", "header")] * 5
    assert header[0].en == "CITY BOARD OF TESTVILLE" and header[0].vi == "HỘI ĐỒNG THÀNH PHỐ THỬ NGHIỆM"
    body = [p for p in pairs if p.anchor not in ("header", "signature")]
    assert {p.anchor for p in body} == {"article:1", "article:2", "article:3"}
    gc = [p for p in body if p.method == "gc"]
    assert [p.link for p in gc if p.link != "1-1"] == ["1-2"]
    (split,) = [p for p in gc if p.link == "1-2"]
    assert split.vi == VI_BODY[4] + " " + VI_BODY[5]
    assert len(split.vi_units) == 2 and len(split.vi_uids) == 2
    assert all(p.method == "zip" for p in body if p.anchor in ("article:1", "article:3"))
    sig = [p for p in pairs if p.anchor == "signature"]
    assert [(p.en, p.vi) for p in sig] == [("CHAIRPERSON", "CHỦ TỊCH"), ("Alex Sample", "Nguyễn Văn Mẫu")]
    assert stats["anchors_matched"] == 3 and stats["chunks"] == 3
    assert sum(stats["dropped_en_chars"].values()) == 0
    assert [p.order for p in pairs] == sorted(p.order for p in pairs)
    assert pairs[0].pair_id == "Synthetic/1#00000" and pairs[0].category == "Test"


def test_one_line_per_link_and_uids_match_units() -> None:
    en, vi = bilingual_pair()
    pairs, _ = align_documents(en, vi, sample_id="Synthetic/1")
    units_en = {u.uid: u.text for u in translation_units(parse_document(en, "en"))}
    units_vi = {u.uid: u.text for u in translation_units(parse_document(vi, "vi"))}
    for p in pairs:
        assert "\n" not in p.en and "\n" not in p.vi
        assert p.en == " ".join(units_en[u] for u in p.en_uids)
        assert p.vi == " ".join(units_vi[u] for u in p.vi_uids)
        json.dumps(p.to_dict(), ensure_ascii=False)


def test_filters_drop_bad_links() -> None:
    en, vi = bilingual_pair()
    vi_bad = vi.replace(VI_BODY[8], "Quy chế này có hiệu lực từ ngày 01 tháng 7 năm 2099.")
    extra = "  Đoạn văn thêm chỉ có ở bản tiếng Việt, không có trong bản tiếng Anh, dài vừa đủ để bị bỏ."
    vi_bad = vi_bad.replace("  " + VI_BODY[1], "  " + VI_BODY[1] + "\n\n" + extra)
    pairs, stats = align_documents(en, vi_bad, sample_id="Synthetic/2", cfg={"keep_dropped": True})
    reasons = Counter(p.drop_reason for p in pairs if p.drop_reason)
    assert reasons["unaligned"] >= 1 or reasons["length_ratio"] >= 1
    kept, _ = align_documents(en, vi_bad, sample_id="Synthetic/2")
    assert all(p.drop_reason is None for p in kept)
    assert len(kept) == len(pairs) - sum(reasons.values())
    assert sum(stats["dropped_vi_chars"].values()) > 0


def test_numeric_filter() -> None:
    en = para_doc(["Article 1. Fees", "The fee is 150,000 for 12 months and 300,000 for 24 months."])
    vi = para_doc(["Điều 1. Lệ phí", "Lệ phí là 999.000 cho 77 tháng và 888.000 cho 66 tháng."])
    pairs, stats = align_documents(en, vi, cfg={"keep_dropped": True})
    (bad,) = [p for p in pairs if p.drop_reason]
    assert bad.drop_reason == "numeric" and bad.num_f1 == 0.0


def test_table_cells_zip_only_when_shapes_match() -> None:
    en_rows = ["    1.    Garden permit            150,000", "    2.    Event permit             1,250,000"]
    vi_rows = ["    1.    Giấy phép vườn hoa       150.000", "    2.    Giấy phép sự kiện        1.250.000"]
    en = "\n".join(["  Article 1. Fees", "", "  The fees are:", ""] + en_rows) + "\n"
    vi = "\n".join(["  Điều 1. Lệ phí", "", "  Lệ phí gồm:", ""] + vi_rows) + "\n"
    pairs, _ = align_documents(en, vi)
    cells = [(p.en, p.vi) for p in pairs if len(p.en_uids) == 1 and "c" in p.en_uids[0]]
    assert ("Garden permit", "Giấy phép vườn hoa") in cells and ("1,250,000", "1.250.000") in cells
    vi_flat = "\n".join(["  Điều 1. Lệ phí", "", "  Lệ phí gồm:", "", vi_rows[0]]) + "\n"
    pairs, stats = align_documents(en, vi_flat, cfg={"keep_dropped": True})
    assert {p.drop_reason for p in pairs if p.method == "table"} == {"table_mismatch"}
    assert stats["table_chunks"] == {"mismatched": 1}


def test_header_columns_when_structure_differs() -> None:
    en = "\n".join([
        "MINISTRY OF GARDENS AND                   REPUBLIC OF EXAMPLELAND",
        "      PUBLIC PARKS                        Order - Progress - Harmony",
        "   No. 12/2031/TT-BVH                          Testville, March 23, 2031",
        "", "  Article 1. Scope", "", "  " + EN_BODY[1][0],
    ]) + "\n"
    vi = "\n".join([
        "BỘ VƯỜN HOA                               CỘNG HÒA XỨ MẪU",
        "   Số: 12/2031/TT-BVH                     Trật tự - Tiến bộ - Hài hòa",
        "                                          Thành phố Thử, ngày 23 tháng 3 năm 2031",
        "", "  Điều 1. Phạm vi điều chỉnh", "", "  " + VI_BODY[1],
    ]) + "\n"
    pairs, stats = align_documents(en, vi)
    assert stats["header_mode"] == "columns"
    got = {(p.en, p.vi) for p in pairs if p.anchor == "header"}
    assert ("MINISTRY OF GARDENS AND PUBLIC PARKS", "BỘ VƯỜN HOA") in got
    assert ("No. 12/2031/TT-BVH", "Số: 12/2031/TT-BVH") in got
    assert ("Testville, March 23, 2031", "Thành phố Thử, ngày 23 tháng 3 năm 2031") in got


def test_nfd_input_aligns_like_nfc() -> None:
    import unicodedata

    en, vi = bilingual_pair()
    a, _ = align_documents(en, vi)
    b, _ = align_documents(en, unicodedata.normalize("NFD", vi))
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]


def test_fit_length_model_recovers_parameters() -> None:
    rng = random.Random(0)
    c_true, s2_true = 0.9, 3.0
    pairs = []
    for _ in range(20000):
        le = rng.randint(20, 600)
        lv = int(round(c_true * le + rng.gauss(0, math.sqrt(s2_true * le))))
        pairs.append((le, max(lv, 1)))
    for _ in range(100):
        pairs.append((rng.randint(20, 60), rng.randint(800, 1200)))
    fit = fit_length_model(pairs)
    assert abs(fit["c"] - c_true) < 0.02
    assert abs(fit["s2"] - s2_true) < 0.4
    assert fit["s2_unclipped"] > fit["s2"]


def test_fit_numeric_weight_and_table() -> None:
    w = fit_numeric_weight([1.0] * 95 + [0.0] * 5)
    assert math.isclose(w["num_weight"], math.log(0.95 / 0.05))
    table = fit_numeric_table(["none"] * 70 + ["f1_ge_08"] * 28 + ["one_sided"] * 2,
                              ["none"] * 50 + ["f1_ge_08"] * 5 + ["one_sided"] * 30 + ["f1_lt_034"] * 15)["num_table"]
    assert table["one_sided"] > 0 and table["f1_lt_034"] > 0 and table["f1_ge_08"] < 0


def test_config_loads_and_resolves() -> None:
    cfg = load_align_config()
    assert cfg["gc"]["num_mode"] in ("table", "either", "both")
    assert 0.8 < cfg["gc"]["c"] < 1.0 and cfg["gc"]["s2"] > 1.43
    assert set(cfg["gc"]["num_table"]) == {"one_sided", "f1_lt_034", "f1_lt_08", "f1_ge_08"}
    same = resolve_cfg({"ratio": 1.0})
    assert same["gc"]["c"] == 1.0 and same["gc"]["s2"] == DEFAULT_ALIGN_CFG["gc"]["s2"]
    assert resolve_cfg(None)["filters"] == DEFAULT_ALIGN_CFG["filters"]


def test_aligned_pair_fields() -> None:
    fields = set(AlignedPair.__dataclass_fields__)
    assert {"pair_id", "sample_id", "category", "order", "en", "vi", "en_units", "vi_units", "link", "method",
            "anchor", "len_ratio", "num_f1", "cost"} <= fields


@pytest.mark.data
def test_alignment_yield_on_real_train_docs() -> None:
    from LegalMT.data import load_split
    from LegalMT.paths import load_config

    dq = load_config("data_quality_v0")
    bad = {sid for ids in dq["broken_pairs"].values() for sid in ids} | {b for _, b in dq["identical_pairs"]}
    df = load_split("train").df
    df = df[~df.sample_id.isin(bad)].reset_index(drop=True)
    rows = random.Random(20260925).sample(range(len(df)), 10)
    kept_en = total_en = 0
    for k in rows:
        r = df.iloc[k]
        pairs, stats = align_documents(r["english_markdown"], r["vietnamese_markdown"], sample_id=r["sample_id"])
        total_en += stats["chars_en"]
        kept_en += sum(len(p.en) for p in pairs)
        assert all("\n" not in p.en and "\n" not in p.vi and p.en and p.vi for p in pairs)
    assert kept_en / total_en >= 0.85


def _brute_force_cost(gc, src, tgt) -> float:
    from functools import lru_cache

    from LegalMT.align import _MOVES

    @lru_cache(maxsize=None)
    def best(i: int, j: int) -> float:
        if i == 0 and j == 0:
            return 0.0
        out = float("inf")
        for di, dj in _MOVES:
            pi, pj = i - di, j - dj
            if pi < 0 or pj < 0:
                continue
            a = sum((u.nums for u in src[pi:i]), Counter()) or None
            b = sum((u.nums for u in tgt[pj:j]), Counter()) or None
            le = sum(len(u.text) for u in src[pi:i])
            lv = sum(len(u.text) for u in tgt[pj:j])
            out = min(out, best(pi, pj) + gc.link_cost((di, dj), le, lv, a, b))
        return out

    return best(len(src), len(tgt))


@pytest.mark.parametrize("seed", range(40))
def test_gale_church_dp_is_optimal_and_covers_both_sides(seed: int) -> None:
    from LegalMT.align import _AUnit, _GC

    rng = random.Random(seed)
    pool = ["12", "345", "2031", "77"]

    def units(k: int) -> list:
        out = []
        for idx in range(k):
            nums = Counter(rng.sample(pool, rng.randint(0, 2))) if rng.random() < 0.5 else Counter()
            out.append(_AUnit("x" * rng.randint(1, 400), idx, f"e{idx}", None, nums))
        return out

    src, tgt = units(rng.randint(0, 6)), units(rng.randint(0, 6))
    for mode in ("table", "either"):
        gc = _GC({**resolve_cfg(None)["gc"], "num_mode": mode})
        links = gc.align(src, tgt)
        i = j = 0
        for i0, i1, j0, j1, _cost in links:
            assert (i0, j0) == (i, j) and (i1 - i0, j1 - j0) in {(1, 1), (1, 0), (0, 1), (2, 1), (1, 2), (2, 2)}
            i, j = i1, j1
        assert (i, j) == (len(src), len(tgt))
        assert math.isclose(sum(c for *_, c in links), _brute_force_cost(gc, src, tgt), rel_tol=1e-9, abs_tol=1e-9)


def test_defaults_mirror_the_yaml() -> None:
    cfg = load_align_config()
    for key in ("gc", "filters", "numbers", "layout"):
        assert cfg[key] == DEFAULT_ALIGN_CFG[key], key


def test_unicode_line_breaks_never_reach_a_pair() -> None:
    odd = [" ", "\x85", "\r", "\x0c", "\x1e", " ", "\x0b"]
    en_lines, vi_lines = ["  Article 1. Scope", ""], ["  Điều 1. Phạm vi", ""]
    for k, ch in enumerate(odd):
        en_lines += [f"  Paragraph {k + 10} first half{ch}second half of the sentence.", ""]
        vi_lines += [f"  Đoạn {k + 10} nửa đầu{ch}nửa sau của câu văn.", ""]
    en_lines += ["NAME ONE" + " " + "X       NAME TWO"]
    vi_lines += ["TÊN MỘT" + " " + "X       TÊN HAI"]
    pairs, _ = align_documents("\n".join(en_lines) + "\n", "\n".join(vi_lines) + "\n", cfg={"keep_dropped": True})
    assert pairs
    for p in pairs:
        for text in (p.en, p.vi):
            assert len(text.splitlines()) <= 1 and text == text.strip()
    for side, lang in ((en_lines, "en"), (vi_lines, "vi")):
        for u in translation_units(parse_document("\n".join(side) + "\n", lang)):
            assert len(u.text.splitlines()) == 1


def test_align_documents_degenerate_inputs() -> None:
    for en, vi in (("", ""), ("  Article 1. Scope\n", ""), ("", "  Điều 1. Phạm vi\n"), ("\n\n", " \n")):
        pairs, stats = align_documents(en, vi, cfg={"keep_dropped": True})
        assert all(p.drop_reason for p in pairs)
        assert stats["kept_links"] == {}
    assert align_elements([], [], {"ratio": 1.0}) == []


def test_align_is_deterministic() -> None:
    en, vi = bilingual_pair()
    a, sa = align_documents(en, vi, sample_id="S/1")
    b, sb = align_documents(en, vi, sample_id="S/1")
    sa.pop("seconds"), sb.pop("seconds")
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b] and sa == sb


def _load_build_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "LegalMT" / "build_aligned.py"
    spec = importlib.util.spec_from_file_location("build_aligned_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_excluded_ids_keep_first_member_of_identical_pairs() -> None:
    mod = _load_build_script()
    dq = {"broken_pairs": {"swapped": ["A/1"], "both_vietnamese": ["B/2", "A/1"], "both_english": None},
          "identical_pairs": [["C/3", "D/4"], ["E/5", "F/6"]]}
    excl = mod.excluded_ids(dq)
    assert excl == {"broken": ["A/1", "B/2"], "duplicate_dropped": ["D/4", "F/6"]}


def test_build_aggregate_handles_empty_and_counts_only_kept() -> None:
    mod = _load_build_script()
    empty = mod.aggregate([])
    assert empty["docs"] == 0 and empty["pairs"] == 0 and empty["kept_en_char_share"] == 0.0
    en, vi = bilingual_pair()
    pairs, stats = align_documents(en, vi, sample_id="S/1", cfg={"keep_dropped": True})
    agg = mod.aggregate([("S/1", [p.to_dict() for p in pairs], stats)])
    assert agg["pairs"] == sum(p.drop_reason is None for p in pairs)
    assert 0.0 < agg["kept_en_char_share"] <= 1.0


def test_build_fingerprint_matches_run_contract() -> None:
    runs = pytest.importorskip("LegalMT.runs")
    from LegalMT.paths import REPO_ROOT

    assert _load_build_script().code_fingerprint(REPO_ROOT) == runs.code_fingerprint(REPO_ROOT)
